#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash потоком на карте 8 ГБ — наш метод (как Nemotron/Flash-Next), заказ helgard 18.09.2026.

Эталонный код DeepSeek (inference/model.py + kernel.py) берётся КАК ЕСТЬ, подменяются только места хранения:
  - маршрутизируемые эксперты (269 ГиБ)  -> читаются по нужде из файлов HF: 2 чтения O_DIRECT на эксперта
                                           (веса w1|w2|w3 одним куском 17.7 МБ + масштабы 1.1 МБ), без перепаковки;
  - таблица Engram (189 ГиБ)             -> строки читаются с диска по хешу (~48 строк на токен);
  - словарь входа и выход (head)          -> в RAM (head fp32 на CPU, как в эталоне — числа те же);
  - wo_a                                  -> хранится FP8 (1.25 ГиБ), разворачивается в bf16 перед умножением
                                           ровно формулой convert.py (эталон держит 2.5 ГиБ bf16).
Всё остальное — на GPU с потолком --vram_gb (эмуляция 8-ГБ карты).

Хранилище экспертов:
  --store ram    все эксперты в RAM (только если хватает — контроль совпадения токенов)
  --store mixed  LRU-кэш в закреплённой RAM на --ram_gb, промахи — с диска (O_DIRECT)
  --store disk   каждый эксперт — с диска
  --emu_gibs X   досыпать чтение до скорости диска X ГиБ/с (Gen5 ≈ 11, наш Gen3 = 2.52 параллельно)
Счёт одинаковый во всех путях ⇒ токены обязаны совпасть.
"""
import argparse, collections, concurrent.futures as cf, json, os, struct, sys, threading, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BLK = 4096
ST = collections.Counter()
TT = collections.defaultdict(float)
EMU = {"gibs": 0.0}
WIN_PIECES = [2, 4]
FAST1 = {"on": False}
PF = {"n": 0, "cand": 12, "count": 0, "touch": 0}
NEXT = {}
_KEEP = []


def pinned_aligned(nbytes):
    """закреплённая память, выровненная по странице (pin_memory() не выровнен — O_DIRECT падает, урок 14.09)"""
    import mmap
    nr = (nbytes + BLK - 1) // BLK * BLK
    mm = mmap.mmap(-1, nr)
    t = torch.from_numpy(np.frombuffer(mm, dtype=np.uint8, count=nbytes))
    rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nr, 0)
    assert int(rc) == 0, f"cudaHostRegister {rc}"
    _KEEP.append(mm)
    return t


# ============================================================ каталог тензоров в файлах HF
class Catalog:
    def __init__(self, d):
        self.d = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.file_of = idx
        self.hdr = {}
        self.meta = {}
        for fn in sorted(set(idx.values())):
            p = os.path.join(d, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            base = 8 + n
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.meta[k] = (p, base + a, b - a, v["dtype"], v["shape"])

    _DT = {"F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8, "BF16": torch.bfloat16, "F32": torch.float32,
           "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "F16": torch.float16, "I32": torch.int32, "I64": torch.int64}

    def tensor(self, name):
        """чтение тензора по смещению из заголовка (02.10 gpu104: safe_open мапит файл ЦЕЛИКОМ —
        Engram-шард 101 ГБ при 31 ГБ RAM даёт ENOMEM)"""
        p, off, nb, dt, shape = self.meta[name]
        buf = bytearray(nb)
        with open(p, "rb", buffering=0) as f:
            f.seek(off)
            mv, got = memoryview(buf), 0
            while got < nb:
                r = f.readinto(mv[got:])
                assert r, f"EOF {name}"
                got += r
        return torch.frombuffer(buf, dtype=torch.uint8).view(self._DT[dt]).reshape(shape)


# ============================================================ эксперты: запись = [веса w1|w2|w3][масштабы w1|w2|w3]
class ExpertStore:
    def __init__(self, cat: Catalog, n_layers, n_exp, mode, ram_gb, threads, chunk):
        self.cat, self.nL, self.E, self.mode = cat, n_layers, n_exp, mode
        m0 = cat.meta["layers.0.ffn.experts.0.w1.weight"]
        s0 = cat.meta["layers.0.ffn.experts.0.w1.scale"]
        self.WB = 3 * m0[2]                     # 17 694 720 байт весов
        self.SB = 3 * s0[2]                     # 1 105 920 байт масштабов
        self.WREG = (self.WB + 2 * BLK - 1) // BLK * BLK + BLK
        self.SREG = (self.SB + 2 * BLK - 1) // BLK * BLK + BLK
        self.REC = self.WREG + self.SREG
        self.shapes = {}
        for w in ("w1", "w2", "w3"):
            self.shapes[w] = (cat.meta[f"layers.0.ffn.experts.0.{w}.weight"][4], cat.meta[f"layers.0.ffn.experts.0.{w}.scale"][4],
                              cat.meta[f"layers.0.ffn.experts.0.{w}.weight"][2], cat.meta[f"layers.0.ffn.experts.0.{w}.scale"][2])
        self.fds = {}
        self.loc = {}
        for li in range(n_layers):
            for e in range(n_exp):
                p = f"layers.{li}.ffn.experts.{e}."
                w1, w2, w3 = (cat.meta[p + f"{w}.weight"] for w in ("w1", "w2", "w3"))
                s1, s2, s3 = (cat.meta[p + f"{w}.scale"] for w in ("w1", "w2", "w3"))
                # проверка раскладки: три веса и три масштаба идут подряд в одном файле
                assert w1[0] == w2[0] == w3[0] == s1[0] and w2[1] == w1[1] + w1[2] and w3[1] == w2[1] + w2[2], p
                assert s2[1] == s1[1] + s1[2] and s3[1] == s2[1] + s2[2], p
                if w1[0] not in self.fds:
                    self.fds[w1[0]] = os.open(w1[0], os.O_RDONLY | os.O_DIRECT)
                self.loc[(li, e)] = (self.fds[w1[0]], w1[1], s1[1])
        self.pool = cf.ThreadPoolExecutor(threads)
        self.opool = cf.ThreadPoolExecutor(1)       # отдельный: fetch сам раздаёт чтения в pool (без взаимной блокировки)
        self.chunk = chunk
        # RAM: кэш (mixed) или всё (ram); плюс кольцо буферов чтения на 2 порции
        self.WIN = WIN_PIECES[0]                     # v2: экспертов в полёте (диск отдаёт их ПО ОЧЕРЕДИ, а не все разом в конце)
        self.PIECES = WIN_PIECES[1]                  # v2: кусков на эксперта (глубина очереди NVMe = WIN×(PIECES+1))
        self.ppool = cf.ThreadPoolExecutor(64)
        self.nbounce = 2 * self.WIN + 2 + PF["n"]     # v4: + строки под предвыборку следующего слоя
        self.pend = {}                               # v4: (слой, эксперт) -> (строка, future) — прочитано заранее, ещё не взято
        self.bfree = collections.deque()   # v2: свободные строки отражения (заполняется после buf)
        self.row_ev = {}
        if mode == "ram":
            n_ram = n_layers * n_exp
        elif mode == "mixed":
            n_ram = int(ram_gb * 2**30 // self.REC)
        else:
            n_ram = 0
        self.n_ram = n_ram
        self.buf = pinned_aligned((n_ram + self.nbounce) * self.REC).view(n_ram + self.nbounce, self.REC)
        self.where, self.lru, self.free = {}, collections.OrderedDict(), list(range(n_ram))
        self.bi = 0
        self.bfree.extend(range(n_ram, n_ram + self.nbounce))
        self.lock = threading.Lock()
        print(f"эксперты: запись {self.REC/2**20:.2f} МиБ, режим {mode}, RAM-кэш {n_ram} записей "
              f"({n_ram*self.REC/2**30:.1f} ГиБ), порция {chunk}", flush=True)
        if mode == "ram":
            t = time.time()
            keys = [(li, e) for li in range(n_layers) for e in range(n_exp)]
            for i, k in enumerate(keys):
                self.where[k] = i
            list(self.pool.map(lambda k: self._read(k, self.where[k]), keys))
            self.free = []
            print(f"все эксперты в RAM за {time.time()-t:.0f} с", flush=True)

    def _read(self, key, row):
        fd, wo, so = self.loc[key]
        r = self.buf[row]
        wa = wo // BLK * BLK
        wl = (wo - wa + self.WB + BLK - 1) // BLK * BLK
        sa = so // BLK * BLK
        sl = (so - sa + self.SB + BLK - 1) // BLK * BLK
        assert wl <= self.WREG and sl <= self.SREG
        mv = memoryview(r.numpy())
        got = os.preadv(fd, [mv[:wl]], wa)          # у конца файла чтение может быть короче блока — это норма
        assert got >= wo - wa + self.WB, (got, wl)
        got = os.preadv(fd, [mv[self.WREG:self.WREG + sl]], sa)
        assert got >= so - sa + self.SB, (got, sl)

    def _read_wait(self, key, row):
        ev = self.row_ev.pop(row, None)
        if ev is not None:
            ev.synchronize()                        # строку ещё копируют на карту — ждём
        t = time.time()
        if self.PIECES <= 1:
            self._read(key, row)
        else:
            fd, wo, so = self.loc[key]
            mv = memoryview(self.buf[row].numpy())
            wa = wo // BLK * BLK; wl = (wo - wa + self.WB + BLK - 1) // BLK * BLK
            sa = so // BLK * BLK; sl = (so - sa + self.SB + BLK - 1) // BLK * BLK
            step = (wl // self.PIECES + BLK - 1) // BLK * BLK
            jobs = [(mv[i:i + min(step, wl - i)], wa + i) for i in range(0, wl, step)]
            jobs.append((mv[self.WREG:self.WREG + sl], sa))
            got = list(self.ppool.map(lambda j: os.preadv(fd, [j[0]], j[1]), jobs))
            assert sum(got[:-1]) >= wo - wa + self.WB and got[-1] >= so - sa + self.SB, (key, got)
        return time.time() - t

    def submit_one(self, li, e, protect):
        """v2: (строка, future|None) — попадание в RAM сразу, промах — своё чтение в пул"""
        k = (li, e)
        with self.lock:
            if k in self.pend:                       # v4: прочитан заранее (возможно, ещё летит)
                r, f = self.pend.pop(k)
                ST["pf_hit"] += 1
                return r, f
            if k in self.where:
                self.lru.move_to_end(k) if k in self.lru else None
                ST["ram_hit"] += 1
                return self.where[k], None
            if self.mode == "mixed":
                if self.free:
                    r = self.free.pop()
                else:
                    victim = next(v for v in self.lru if v not in protect and v not in self.pend)
                    r = self.where.pop(victim); del self.lru[victim]
                self.where[k] = r; self.lru[k] = None
            else:
                # ⛔#320 гонка: кольцо по кругу перезаписывало строку, ещё НЕ отданную карте (29/64 слов). Теперь — список свободных:
                # строка возвращается в release() после постановки перекачки; читатель ждёт её событие перед записью.
                assert self.bfree, "нет свободной строки отражения — окно больше кольца"
                r = self.bfree.popleft()
            ST["disk_read"] += 1; ST["disk_bytes"] += self.WB + self.SB
        return r, self.pool.submit(self._read_wait, k, r)

    def prefetch(self, li, e, protect):
        """v4: заранее прочитать эксперта следующего слоя. False — уже есть (RAM/летит) или нет строки."""
        k = (li, e)
        with self.lock:
            if k in self.where or k in self.pend:
                if PF["touch"] and k in self.lru:
                    self.lru.move_to_end(k)          # --pf_touch (моя добавка, отдельно от идеи helgard): освежить по прогнозу
                ST["pf_cached"] += 1
                return "cached"
            if self.mode == "mixed":
                if self.free:
                    r = self.free.pop()
                else:
                    victim = next(v for v in self.lru if v not in protect and v not in self.pend)
                    r = self.where.pop(victim); del self.lru[victim]
                self.where[k] = r; self.lru[k] = None
            else:
                if not self.bfree:
                    return False
                r = self.bfree.popleft()
            ST["pf_read"] += 1; ST["disk_bytes"] += self.WB + self.SB
        self.pend[k] = (r, self.pool.submit(self._read_wait, k, r))
        return True

    def drop_pf(self, li):
        """v4: неугаданные заранее прочитанные эксперты слоя li — дождаться чтения и вернуть строку
        (в mixed запись остаётся в кэше как обычная — её чтение к этому моменту закончено)"""
        for k in [k for k in self.pend if k[0] == li]:
            r, f = self.pend.pop(k)
            f.result()
            ST["pf_waste"] += 1
            self.release(r)

    def busy(self):
        return sum(1 for _, f in self.pend.values() if not f.done())

    def release(self, row):
        if row >= self.n_ram:
            with self.lock:
                self.bfree.append(row)

    def can_read(self):
        return self.mode != "disk" or len(self.bfree) > 0

    def deltas(self, key):
        _, wo, so = self.loc[key]
        return wo - wo // BLK * BLK, self.WREG + (so - so // BLK * BLK)

    def fetch(self, li, experts, protect):
        """{e: строка pinned} — из RAM-кэша или с диска (параллельно). protect — не вытеснять эти ключи."""
        out, to_read = {}, []
        with self.lock:
            for e in experts:
                k = (li, e)
                if k in self.where:
                    self.lru.move_to_end(k) if k in self.lru else None
                    out[e] = self.where[k]; ST["ram_hit"] += 1
                else:
                    to_read.append(e)
            rows = []
            for e in to_read:
                k = (li, e)
                if self.mode == "mixed":
                    if self.free:
                        r = self.free.pop()
                    else:
                        victim = next(v for v in self.lru if v not in protect)
                        r = self.where.pop(victim); del self.lru[victim]
                    self.where[k] = r; self.lru[k] = None
                else:
                    r = self.n_ram + self.bi; self.bi = (self.bi + 1) % self.nbounce
                rows.append((k, r))
                out[e] = r
        if rows:
            t = time.time()
            list(self.pool.map(lambda kr: self._read(*kr), rows))
            if EMU["gibs"]:
                nb = len(rows) * (self.WB + self.SB)
                lag = nb / (EMU["gibs"] * 2**30) - (time.time() - t)
                if lag > 0:
                    time.sleep(lag)
            TT["disk_s"] += time.time() - t
            ST["disk_read"] += len(rows); ST["disk_bytes"] += len(rows) * (self.WB + self.SB)
        return out


class GpuStage:
    """v2 (#320, схема Codex): кольцо ячеек на GPU, копирование в ОТДЕЛЬНОМ потоке, события вместо synchronize.
    Ячейку перезаписываем только после события «счёт по ней закончен»; строку RAM — после события «скопирована»."""
    def __init__(self, store, nslots):
        self.s, self.N = store, nslots
        self.GREC = (store.WB + store.SB + 255) // 256 * 256
        self.g = torch.empty((nslots, self.GREC), dtype=torch.uint8, device="cuda")
        self.cs = torch.cuda.Stream()
        self.free_ev = [None] * nslots
        self.k = 0

    def next_slot(self):
        sl = self.k; self.k = (self.k + 1) % self.N; return sl

    def load(self, slot, row, key):
        dw, ds = self.s.deltas(key)
        host = self.s.buf[row]
        dst = self.g[slot]
        if self.free_ev[slot] is not None:
            self.cs.wait_event(self.free_ev[slot])
        with torch.cuda.stream(self.cs):
            dst[:self.s.WB].copy_(host[dw:dw + self.s.WB], non_blocking=True)
            dst[self.s.WB:self.s.WB + self.s.SB].copy_(host[ds:ds + self.s.SB], non_blocking=True)
            ev = torch.cuda.Event(); ev.record(self.cs)
        self.s.row_ev[row] = ev
        return ev

    def done(self, slot):
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream()); self.free_ev[slot] = ev

    def views(self, slot):
        g = self.g[slot]
        out, wo, so = {}, 0, self.s.WB
        for w in ("w1", "w2", "w3"):
            wshape, sshape, wb, sb = self.s.shapes[w]
            wt = g[wo:wo + wb].view(torch.float4_e2m1fn_x2).view(*wshape)
            wt.scale = g[so:so + sb].view(torch.float8_e8m0fnu).view(*sshape)
            out[w] = wt
            wo += wb; so += sb
        return out


# ============================================================ сборка модели на эталонном коде
def build(a):
    sys.path.insert(0, a.code)
    import model as M
    from kernel import act_quant  # noqa: F401 (проверка, что ядра импортируются)

    cat = Catalog(a.dir)
    cfg = json.load(open(os.path.join(a.code, "config.json")))
    args = M.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = a.max_seq
    args.temperature = 0.0
    args.dspark_block_size = 0          # черновые слои не грузим
    args.vision_n_layers = 0            # зрение не грузим (текстовый замер)

    WIN_PIECES[:] = [a.win, a.pieces]
    PF["n"] = getattr(a, "pf", 0)
    PF["count"] = getattr(a, "pf_count", 0)
    PF["touch"] = getattr(a, "pf_touch", 0)
    FAST1["on"] = bool(getattr(a, "fast1", 0))
    store = ExpertStore(cat, args.n_layers, args.n_routed_experts, a.store, a.ram_gb, a.threads, a.chunk)
    stage = GpuStage(store, 2 * store.WIN + 2)

    def expert_fwd(x, w, weights, limit):
        gate = M.linear(x, w["w1"]).float()
        up = M.linear(x, w["w3"]).float()
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate = torch.clamp(gate, max=limit)
        h = F.silu(gate) * up
        h = weights * h
        return M.linear(h.to(x.dtype), w["w2"])

    class StreamMoE(nn.Module):
        def __init__(self, layer_id, args):
            super().__init__()
            self.layer_id, self.dim = layer_id, args.dim
            self.n_routed_experts = args.n_routed_experts
            self.gate = M.Gate(layer_id, args)
            self.shared_experts = M.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
            self.limit = args.swiglu_limit

        def forward(self, x, image_mask=None):
            shape = x.size()
            x = x.view(-1, self.dim)
            weights, indices = self.gate(x, None)
            y = torch.zeros_like(x, dtype=torch.float32)
            one = FAST1["on"] and x.size(0) == 1
            pf_list = collections.deque()
            if one:
                # v3 (#321, Codex п.3): один токен — индексы на CPU ОДНОЙ синхронизацией на слой; дальше без torch.where
                # (nonzero останавливал CPU до конца копирования КАЖДОГО эксперта: 240 остановок на токен)
                nx = NEXT.get(self.layer_id) if PF["n"] else None
                if nx is not None:
                    # v4 (#321): прогноз A из route_probe (71% топ-6 на 1 слой): снять ffn_norm_i, надеть ffn_norm_{i+1}, gate_{i+1};
                    # та же синхронизация, что и индексы
                    g2, ratio = nx
                    sc = M.linear((x.float() * ratio), g2.weight.float()) / g2.gate_temp
                    sc = F.softplus(sc).sqrt() if g2.score_func not in ("softmax", "sigmoid") else (sc.softmax(-1) if g2.score_func == "softmax" else sc.sigmoid())
                    pred = (sc + g2.bias).topk(PF["cand"], dim=-1)[1]
                    both = torch.cat([indices[0], pred[0]]).tolist()
                    row, pf_list = both[:indices.size(1)], collections.deque(both[indices.size(1):])
                else:
                    row = indices[0].tolist()
                uniq = sorted(set(row))
                topof = {e: row.index(e) for e in uniq}
            else:
                counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
                uniq = [i for i in range(self.n_routed_experts) if counts[i]]      # по возрастанию — как эталон
            ST["experts_used"] += len(uniq)
            protect = set((self.layer_id, e) for e in uniq)
            ys = self.shared_experts(x)                     # v2: общий эксперт — в очередь карты сразу, прибавим последним
            outs = {}
            todo = collections.deque(uniq)
            ready, inflight = [], {}
            def fill():
                while todo and len(inflight) < store.WIN and store.can_read():
                    e = todo.popleft()
                    r, f = store.submit_one(self.layer_id, e, protect)
                    if f is None:
                        ready.append((e, r))
                    else:
                        inflight[f] = (e, r)
            pf_done = [0]
            def pf_fill(final=False):
                # диск свободен от настоящих чтений этого слоя ⇒ подвозим P самых вероятных экспертов следующего слоя
                while pf_list and pf_done[0] < PF["n"] and not todo and (final or len(inflight) + store.busy() < store.WIN):
                    if store.mode == "disk" and not store.bfree:
                        break
                    r = store.prefetch(self.layer_id + 1, pf_list.popleft(), protect)
                    if r is True or (r == "cached" and PF["count"]):
                        pf_done[0] += 1                   # pf_count: кандидат из RAM тоже тратит бюджет P (не читаем глубже)
            fill()
            while ready or inflight:
                if not ready:
                    t = time.time()
                    dn, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
                    TT["wait_s"] += time.time() - t
                    for f in dn:
                        TT["disk_s"] += f.result()
                        ready.append(inflight.pop(f))
                e, r = ready.pop(0)
                slot = stage.next_slot()
                ev = stage.load(slot, r, (self.layer_id, e))
                store.release(r)
                ST["h2d_bytes"] += store.WB + store.SB
                fill()
                pf_fill()
                torch.cuda.current_stream().wait_event(ev)
                if one:
                    t = topof[e]
                    outs[e] = (None, expert_fwd(x, stage.views(slot), weights[:, t:t + 1], self.limit))
                else:
                    idx, top = torch.where(indices == e)
                    outs[e] = (idx, expert_fwd(x[idx], stage.views(slot), weights[idx, top, None], self.limit))
                stage.done(slot)
            for e in uniq:                                  # сложение в порядке эталона (по возрастанию номера)
                idx, o = outs[e]
                if idx is None:
                    y += o
                else:
                    y[idx] += o
            if PF["n"]:
                store.drop_pf(self.layer_id)                # неугаданные для этого слоя — все свои эксперты уже взяты
            pf_fill(final=True)
            y += ys
            return y.type_as(x).view(shape)

    class DiskEngramEmbedding(nn.Module):
        def __init__(self, num_embeddings, dim):
            super().__init__()
            self.dim, self.block_size = dim, M.fp8_block_size
            self.name = None

        def bind(self, name):
            self.wmeta = cat.meta[name + ".weight"]
            self.smeta = cat.meta[name + ".scale"]
            self.fd = os.open(self.wmeta[0], os.O_RDONLY)
            self.rw = self.wmeta[4][1]                          # 256 байт fp8 на строку
            self.rs = self.smeta[4][1]                          # 8 байт e8m0 на строку

        def forward(self, indices):
            t = time.time()
            ids = indices.flatten().tolist()
            u = sorted(set(ids))
            def rd(i):
                return (os.pread(self.fd, self.rw, self.wmeta[1] + i * self.rw),
                        os.pread(self.fd, self.rs, self.smeta[1] + i * self.rs))
            got = list(store.pool.map(rd, u))
            pos = {i: k for k, i in enumerate(u)}
            W = torch.frombuffer(bytearray(b"".join(g[0] for g in got)), dtype=torch.uint8).view(len(u), self.rw)
            S = torch.frombuffer(bytearray(b"".join(g[1] for g in got)), dtype=torch.uint8).view(len(u), self.rs)
            sel = torch.tensor([pos[i] for i in ids], device="cpu")
            W = W[sel].view(torch.float8_e4m3fn).cuda().view(*indices.shape, self.rw)
            S = S[sel].view(torch.float8_e8m0fnu).cuda().view(*indices.shape, self.rs)
            v = W.float().unflatten(-1, (-1, self.block_size)) * S.float().unsqueeze(-1)
            ST["engram_rows"] += len(u)
            TT["engram_s"] += time.time() - t
            return v.flatten(-2).to(torch.bfloat16)

    class CPUEmbedding(nn.Module):
        def __init__(self, vocab_size, dim):
            super().__init__()
            self.w = None

        def forward(self, x):
            return F.embedding(x.cpu(), self.w).cuda()

    class CPUHead(nn.Module):
        """как эталонный ParallelHead: fp32, только последняя позиция — но на CPU (2.47 ГиБ вне карты)"""
        def __init__(self, vocab_size, dim, norm_eps=1e-6, hc_eps=1e-6):
            super().__init__()
            self.w = None

        def forward(self, x, full_logits=False):
            if not full_logits:
                x = x[:, -1]
            t = time.time()
            r = F.linear(x.float().cpu(), self.w)
            TT["head_s"] += time.time() - t
            return r

    class WoA(nn.Module):
        """wo_a хранится FP8 + масштабы 32×32; .weight разворачивает в bf16 формулой convert.py"""
        def __init__(self):
            super().__init__()
            self.qweight = None
            self.qscale = None

        @property
        def weight(self):
            w, s = self.qweight, self.qscale
            ob, ib = w.size(0) // s.size(0), w.size(1) // s.size(1)
            # convert.py считает во float32 и округляет в bf16; здесь сразу bf16 — ТО ЖЕ до бита:
            # fp8 e4m3 (3 бита мантиссы) × степень двойки (e8m0) точно представимо в bf16 (7 бит), без 128 МБ float32
            x = w.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).to(torch.bfloat16) * s[:, None, :, None].to(torch.bfloat16)
            return x.flatten(2, 3).flatten(0, 1)

    # ★sparse_attn эталона держит в общей памяти все 64 головы: 141 КБ на блок, а у игровых Blackwell (sm_120,
    # и 5090, и наша 5060) предел ~99 КБ ⇒ "Failed to set the allowed dynamic shared memory size to 141312".
    # Головы в этом внимании независимы (свой softmax у каждой) ⇒ зовём ТО ЖЕ ядро группами по --attn_heads голов:
    # формулы те же, меняется только сколько голов живёт в блоке одновременно.
    _orig_sparse_attn = M.sparse_attn

    ATT_CHUNK = int(os.environ.get("DS_ATT_CHUNK", "256"))

    def sparse_attn_grouped(q, kv, attn_sink, topk_idxs, softmax_scale):
        # #320: ядро считает каждую позицию запроса своим блоком (T.Kernel(m, b)), индексы [b, m, topk] ⇒
        # резка по позициям бит-в-бит; выход пишем в готовый тензор (torch.cat удваивал память: промт ~1350 ток = OOM)
        b, m, h = q.shape[:3]
        g = a.attn_heads
        if h <= g and m <= ATT_CHUNK:
            return _orig_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
        out = torch.empty_like(q)
        for i in range(0, h, g):
            sink = attn_sink[i:i + g].contiguous()
            for j in range(0, m, ATT_CHUNK):
                qq = q[:, j:j + ATT_CHUNK, i:i + g].clone(memory_format=torch.contiguous_format)
                ti = topk_idxs[:, j:j + ATT_CHUNK].contiguous() if topk_idxs.size(1) == m else topk_idxs
                out[:, j:j + ATT_CHUNK, i:i + g] = _orig_sparse_attn(qq, kv, sink, ti, softmax_scale)
        return out
    M.sparse_attn = sparse_attn_grouped

    # #320: склейки гиперсвязей (hc_mixes/hc_pre/hc_post) считаются по каждой позиции отдельно ⇒ на длинном промте
    # режем по позициям кусками HC_CHUNK: то же число, пик памяти в разы меньше (промт 650 ток падал по VRAM, 02.10)
    HC_CHUNK = int(os.environ.get("DS_HC_CHUNK", "128"))
    _hm, _hp, _ho = M.Block.hc_mixes, M.Block.hc_pre, M.Block.hc_post

    def hc_mixes_c(self, x, fn, sc, base):
        if x.size(1) <= HC_CHUNK:
            return _hm(self, x, fn, sc, base)
        parts = [_hm(self, x[:, i:i + HC_CHUNK], fn, sc, base) for i in range(0, x.size(1), HC_CHUNK)]
        return tuple(torch.cat(t, dim=1) for t in zip(*parts))

    def hc_pre_c(self, x, pre_mix):
        if x.size(1) <= HC_CHUNK:
            return _hp(self, x, pre_mix)
        return torch.cat([_hp(self, x[:, i:i + HC_CHUNK], pre_mix[:, i:i + HC_CHUNK]) for i in range(0, x.size(1), HC_CHUNK)], dim=1)

    def hc_post_c(self, x, residual, post, comb):
        if x.size(1) <= HC_CHUNK:
            return _ho(self, x, residual, post, comb)
        return torch.cat([_ho(self, x[:, i:i + HC_CHUNK], residual[:, i:i + HC_CHUNK], post[:, i:i + HC_CHUNK], comb[:, i:i + HC_CHUNK])
                          for i in range(0, x.size(1), HC_CHUNK)], dim=1)
    M.Block.hc_mixes, M.Block.hc_pre, M.Block.hc_post = hc_mixes_c, hc_pre_c, hc_post_c

    M.MoE = StreamMoE
    M.ParallelEngramEmbedding = DiskEngramEmbedding
    M.ParallelEmbedding = CPUEmbedding
    M.ParallelHead = CPUHead
    _orig_attn_init = M.Attention.__init__

    def attn_init(self, layer_id, args):
        _orig_attn_init(self, layer_id, args)
        self.wo_a = WoA()                    # bf16-заготовку эталона (64 МБ на слой) сразу отпускаем
    M.Attention.__init__ = attn_init

    if a.vram_gb:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 2**30 / tot))
        print(f"потолок VRAM {a.vram_gb} ГиБ из {tot/2**30:.1f}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.dir)
    torch.set_default_dtype(torch.bfloat16)
    t = time.time()
    with torch.device("cuda"):
        model = M.Transformer(args, tok)
    print(f"каркас за {time.time()-t:.0f} с, VRAM {torch.cuda.memory_allocated()/2**30:.2f} ГиБ", flush=True)

    # --- загрузка весов (имена HF уже в формате эталона; правила convert.py для mp=1)
    params = dict(model.named_parameters())
    mods = dict(model.named_modules())
    loaded, skipped = 0, collections.Counter()
    got = set()
    t = time.time()
    for name in sorted(cat.meta):
        n = name[len("model."):] if name.startswith("model.") else name
        if n.startswith("mtp.") or n.startswith("vision.") or n.startswith("aligner.") or n.startswith("image_"):
            skipped["mtp/vision"] += 1; continue
        n = n.replace("self_attn", "attn").replace("mlp", "ffn").replace("weight_scale_inv", "scale") \
             .replace("e_score_correction_bias", "bias")
        if ".ffn.experts." in n:
            skipped["experts(stream)"] += 1; continue
        if ".engram.embed." in n:
            mod = mods[n.rsplit(".", 1)[0]]
            if n.endswith(".weight"):
                mod.bind(name.rsplit(".", 1)[0])
            skipped["engram(disk)"] += 1; continue
        if n == "embed.weight":
            model.embed.w = cat.tensor(name).to(torch.bfloat16); loaded += 1; continue
        if n == "head.weight":
            model.head.w = cat.tensor(name).float(); loaded += 1; continue
        if ".attn.wo_a." in n:
            mod = mods[n.rsplit(".", 1)[0]]
            v = cat.tensor(name).cuda()
            if n.endswith(".weight"):
                mod.qweight = v
            else:
                mod.qscale = v
            loaded += 1; continue
        if n not in params:
            skipped["нет в модели: " + n.split(".")[-1]] += 1
            if skipped["нет в модели: " + n.split(".")[-1]] <= 2:
                print("  нет параметра", n, flush=True)
            continue
        p = params[n]
        v = cat.tensor(name)
        assert tuple(v.shape) == tuple(p.shape), (n, v.shape, p.shape)
        with torch.no_grad():
            if p.dtype != v.dtype and v.dtype in (torch.int8, torch.uint8):
                v = v.view(p.dtype)
            p.copy_(v.to(p.device))
        got.add(n)
        loaded += 1
    missing = [k for k in params if k not in got]
    print(f"НЕ ЗАГРУЖЕНЫ параметры модели: {len(missing)} {missing[:8]}", flush=True)
    assert not missing, "есть параметры без весов — числа были бы мусором"
    if PF["n"]:
        L = model.layers
        for i in range(len(L) - 1):
            NEXT[i] = (L[i + 1].ffn.gate, (L[i + 1].ffn_norm.weight.float() / L[i].ffn_norm.weight.float()))
        print(f"предвыборка: P={PF['n']} кандидатов {PF['cand']}, слоёв {len(NEXT)}", flush=True)
    print(f"веса: загружено {loaded}, пропущено {dict(skipped)} за {time.time()-t:.0f} с; "
          f"VRAM {torch.cuda.memory_allocated()/2**30:.2f} ГиБ", flush=True)
    return model, tok, args


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="папка с файлами HF")
    ap.add_argument("--code", required=True, help="папка inference/ из репо (эталон)")
    ap.add_argument("--store", choices=["ram", "mixed", "disk"], required=True)
    ap.add_argument("--ram_gb", type=float, default=20)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--vram_gb", type=float, default=7.3)
    ap.add_argument("--max_seq", type=int, default=4096)
    ap.add_argument("--attn_heads", type=int, default=16, help="голов на вызов sparse_attn (sm_120: общая память ~99 КБ)")
    ap.add_argument("--emu_gibs", type=float, default=0.0)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--prefill_len", type=int, default=300)
    ap.add_argument("--out", default="v41_result.json")
    ap.add_argument("--win", type=int, default=2, help="v2: экспертов в полёте")
    ap.add_argument("--pieces", type=int, default=4, help="v2: кусков на эксперта")
    ap.add_argument("--fast1", type=int, default=0, help="v3: короткий путь одного токена (без torch.where)")
    ap.add_argument("--pf_count", type=int, default=0, help="v4: кандидат, уже лежащий в RAM, засчитывается в P (идея helgard)")
    ap.add_argument("--pf_touch", type=int, default=0, help="v4: кандидат из RAM поднимается в LRU по прогнозу (отдельный опыт)")
    ap.add_argument("--pf", type=int, default=0, help="v4: сколько экспертов следующего слоя читать заранее (нужен --fast1)")
    return ap


if __name__ == "__main__":
    a = parser().parse_args()
    EMU["gibs"] = a.emu_gibs
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model, tok, args = build(a)
    torch.set_default_device("cuda")          # как generate.py эталона: служебные тензоры внимания создаются на карте
    sys.path.insert(0, os.path.join(a.code, "..", "encoding"))
    from encoding import encode_messages

    def chat(text):
        return tok.encode(encode_messages([{"role": "user", "content": text}], thinking_mode="chat"))

    @torch.inference_mode()
    def gen(ids, new):
        ST.clear(); TT.clear()
        torch.cuda.synchronize(); t0 = time.time()
        out_ids, _, _ = model.forward(torch.tensor([ids], device="cuda"), 0)
        torch.cuda.synchronize(); t_pre = time.time() - t0
        pre = (dict(ST), {k: round(v, 3) for k, v in TT.items()})
        ST.clear(); TT.clear()
        g = [int(out_ids[0])]
        pos = len(ids)
        t1 = time.time()
        while len(g) < new and g[-1] != tok.eos_token_id:
            out_ids, _, _ = model.forward(torch.tensor([[g[-1]]], device="cuda"), pos)
            pos += 1
            g.append(int(out_ids[0]))
        torch.cuda.synchronize()
        dt = time.time() - t1
        return g, t_pre, dt, pre, (dict(ST), {k: round(v, 3) for k, v in TT.items()})

    res = {"args": vars(a), "runs": []}
    gen(chat("Say hello."), 3)           # прогрев: сборка ядер TileLang вне замера
    prompts = ["Назови столицу Австралии одним словом.", "What is the capital of Australia? Answer briefly.",
               "Explain the difference between TCP and UDP and give one use case for each.",
               "Write a Python function that merges two sorted lists into one sorted list."]
    for p in prompts:
        ids = chat(p)
        new = 16 if len(p) < 50 else a.new
        g, t_pre, dt, pre, dec = gen(ids, new)
        nd = max(1, len(g) - 1)
        r = {"prompt": p[:50], "prompt_tok": len(ids), "gen_tok": len(g), "prefill_s": round(t_pre, 2),
             "decode_s": round(dt, 2), "tok_s": round(nd / dt, 3), "ids": g,
             "text": tok.decode(g, skip_special_tokens=True)[:300], "prefill_stats": pre, "decode_stats": dec,
             "per_tok": {k: round(v / nd, 4) for k, v in dec[1].items()},
             "disk_mib_per_tok": round(dec[0].get("disk_bytes", 0) / nd / 2**20, 1)}
        res["runs"].append(r)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "ids"}, ensure_ascii=False), flush=True)
    # длинный промт: время до первого токена
    gpl = open("/usr/share/common-licenses/GPL-3").read() if os.path.exists("/usr/share/common-licenses/GPL-3") else "lorem ipsum " * 400
    pid = tok.encode(gpl)[: a.prefill_len]
    ids = chat(tok.decode(pid) + "\n\nSummarize the text above.")
    g, t_pre, dt, pre, dec = gen(ids, 1)
    res["prefill"] = {"prompt_tok": len(ids), "sec": round(t_pre, 2), "stats": pre}
    print("PREFILL", json.dumps(res["prefill"], ensure_ascii=False), flush=True)
    res["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"DONE пик VRAM {res['peak_vram_gib']:.2f} ГиБ", flush=True)
