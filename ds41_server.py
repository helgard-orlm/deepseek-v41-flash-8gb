#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash потоком с /fast — служба с API как у OpenAI (/v1/models, /v1/chat/completions со стримом)
для Open WebUI (:9800). Заказ helgard 02.10.2026 «давай в open webui».

Движок — v41_stream_v2.py (#320 02.10: 1.28 ток/с только с диска, было 0.89), без изменений; здесь только обёртка:
  - модель грузится один раз при старте службы (~20 с), карта занята целиком (7.3 ГиБ) ⇒ запуск из панели gpusvcs;
  - каждый запрос = весь диалог заново (снимков разговора пока нет): промт читается с позиции 0;
  - жадный выбор слова (args.temperature = 0 в движке) — как чат Nemotron;
  - обрыв соединения (кнопка «стоп» в чате) останавливает генерацию после текущего слова.
"""
import json, os, sys, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("DS_PORT", "9804"))
MODEL_DIR = os.environ.get("DS_DIR", "/fast/DeepSeek-V4.1-Flash")
CODE_DIR = os.environ.get("DS_CODE", os.path.join(HERE, "inference_fix"))   # эталон + фикс гонки act_quant (02.10)
RAM_GB = os.environ.get("DS_RAM_GB", "12")
MAX_SEQ = int(os.environ.get("DS_MAX_SEQ", "8192"))
DEFAULT_MAX_TOKENS = int(os.environ.get("DS_MAX_TOKENS", "2048"))
MODEL_ID = "deepseek-v4.1-flash"

STATE = {"status": "loading", "error": None, "since": time.time(), "served": 0}
ENGINE = {}
GEN_LOCK = threading.Lock()

# ---- #320 (02.10, просьба helgard): снимки разговора — не перечитывать уже прочитанное начало
# Состояние модели целиком = буферы (KV внимания, кольцо окна 128, хвосты сжатия, кэш индексатора, история токенов Engram),
# все по позициям ⇒ снимок «после N токенов» + дочитать хвост ПО ОДНОМУ токену (эталон не умеет пачку с start_pos>0).
PREFIX = os.environ.get("DS_PREFIX_CACHE", "1") == "1"
SNAP_N = int(os.environ.get("DS_SNAP_N", "8"))                      # снимков в RAM
SNAP_DIR = os.environ.get("DS_SNAP_DIR", os.path.expanduser("~/.cache/ds41_snaps"))  # копия на диск (переживает перезапуск); "" = нет; /fast не под кэши (слово helgard)
PREFILL_MAX = int(os.environ.get("DS_PREFILL_MAX", "700"))          # #320 замер: пачкой 781 ток ок, ~1000 = OOM (запас VRAM ~430 МБ) ⇒ остаток по одному
TOK_DECODE_S = float(os.environ.get("DS_TOK_DECODE_S", "0.75"))      # цена 1 токена хвоста (замер 02.10: 0.6–0.8 с)
PRE_FIXED_S, PRE_TOK_S = 4.0, 0.045                                 # цена полного чтения ≈ 4 с + 0.045 с/ток (18 ток 5.9 с, 311 ток 17.9 с)
STATE_BUF = ("kv_state", "score_state", "k_cache", "window_kv_cache", "compress_kv_cache", "cache")
SNAPS = __import__("collections").OrderedDict()                       # ключ -> (fed: tuple, {буфер: cpu})
LIVE = {"fed": None}                                                  # какие токены сейчас в буферах модели
SNAP_VER = {"v": ""}


def cost_scratch(n):
    k = min(n, PREFILL_MAX)
    return PRE_FIXED_S + PRE_TOK_S * k + TOK_DECODE_S * (n - k)


def state_bufs(model):
    return [(n, b) for n, b in model.named_buffers() if n.rsplit(".", 1)[-1] in STATE_BUF]


def snap_take(model, fed):
    import hashlib, torch
    st = {n: b.detach().to("cpu", copy=True) for n, b in state_bufs(model)}
    key = hashlib.sha1((SNAP_VER["v"] + repr(fed)).encode()).hexdigest()[:16]
    SNAPS[key] = (tuple(fed), st); SNAPS.move_to_end(key)
    while len(SNAPS) > SNAP_N:
        old, _ = SNAPS.popitem(last=False)
        if SNAP_DIR:
            try: os.remove(os.path.join(SNAP_DIR, old + ".pt"))
            except OSError: pass
    if SNAP_DIR:
        tmp = os.path.join(SNAP_DIR, key + ".tmp")
        torch.save({"ver": SNAP_VER["v"], "fed": list(fed), "st": st}, tmp); os.replace(tmp, os.path.join(SNAP_DIR, key + ".pt"))


def snap_restore(model, st):
    with __import__("torch").no_grad():
        for n, b in state_bufs(model):
            b.copy_(st[n], non_blocking=False)


def snap_load_disk():
    import glob, torch
    if not SNAP_DIR: return
    os.makedirs(SNAP_DIR, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")), key=os.path.getmtime)[-SNAP_N:]
    n = 0
    for f in fs:
        try:
            d = torch.load(f, map_location="cpu")
            if d.get("ver") != SNAP_VER["v"]: continue                  # другой движок/код ⇒ снимок чужой
            SNAPS[os.path.basename(f)[:-3]] = (tuple(d["fed"]), d["st"]); n += 1
        except Exception as e:
            log("снимок не прочитан", f, e)
    log(f"снимков с диска: {n}")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_ollama():
    """модели ollama делят ту же карту 8 ГБ — выгружаем перед загрузкой (как nem_server)"""
    try:
        ps = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3).read())
        for m in ps.get("models", []):
            rq = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                        data=json.dumps({"model": m["name"], "keep_alive": 0}).encode(),
                                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(rq, timeout=30).read()
            log("ollama выгружена:", m["name"])
    except Exception as e:
        log("ollama: пропуск выгрузки:", e)


def load_engine():
    try:
        free_ollama()
        sys.path.insert(0, HERE)
        import torch
        import v41_stream_v4 as V                  # #321: v3 + предвыборка 2 экспертов след. слоя (--pf 2): 1.64 диск / 2.3 RAM14, слова = эталон
        a = V.parser().parse_args(["--dir", MODEL_DIR, "--code", CODE_DIR, "--fast1", "1", "--pf", "2", "--store", "mixed", "--ram_gb", RAM_GB,
                                   "--max_seq", str(MAX_SEQ)])
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        torch.set_num_threads(8)
        torch.manual_seed(0)
        model, tok, _ = V.build(a)
        torch.set_default_device("cuda")          # как в v41_stream: служебные тензоры внимания на карте
        sys.path.insert(0, os.path.join(a.code, "..", "encoding"))
        from encoding import encode_messages
        ENGINE.update(model=model, tok=tok, enc=encode_messages, V=V, torch=torch)
        import hashlib
        h = hashlib.sha1(str(MAX_SEQ).encode())
        for f in ("model.py", "kernel.py", "engram.py"):
            h.update(open(os.path.join(CODE_DIR, f), "rb").read())
        h.update(open(V.__file__, "rb").read())
        SNAP_VER["v"] = h.hexdigest()[:12]
        mb = sum(b.numel() * b.element_size() for _, b in state_bufs(model)) / 2**20
        log(f"снимки: версия {SNAP_VER['v']}, состояние {mb:.1f} МиБ, в RAM {SNAP_N}, диск {SNAP_DIR or 'нет'}")
        snap_load_disk()
        STATE["status"] = "ready"
        log(f"готово: VRAM {torch.cuda.memory_allocated() / 2**30:.2f} ГиБ, RAM-кэш экспертов {RAM_GB} ГБ, max_seq {MAX_SEQ}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        STATE.update(status="error", error=repr(e))


def conv_msg(m):
    """сообщение OpenAI -> формат encode_messages (только текст; картинки/инструменты не поддержаны)"""
    c = m.get("content")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    role = m.get("role", "user")
    if role not in ("system", "user", "assistant"):
        role = "user"
    return {"role": role, "content": c or ""}


def generate(messages, max_tokens, on_text, cancelled):
    torch, model, tok = ENGINE["torch"], ENGINE["model"], ENGINE["tok"]
    ids = tok.encode(ENGINE["enc"](messages, thinking_mode="chat"))
    if len(ids) + max_tokens > MAX_SEQ:
        max_tokens = MAX_SEQ - len(ids)
        if max_tokens < 16:
            on_text(f"[диалог слишком длинный: {len(ids)} токенов при пределе {MAX_SEQ} — начните новый чат]")
            return 0, len(ids), 0.0, 0.0
    g, shown = [], ""
    # set_default_device в потоке загрузки на этот поток не действует (он потоковый) ⇒ задаём здесь
    with torch.inference_mode(), torch.device("cuda"):
        torch.cuda.synchronize(); t0 = time.time()
        start, src = 0, "с нуля"
        if PREFIX and not STATE.get("nocache"):
            cands = [("живое", LIVE["fed"], None)] + [("снимок", f, st) for f, st in reversed(list(SNAPS.values()))]
            best = None
            for name, fed, st in cands:
                if fed and len(fed) < len(ids) and tuple(ids[:len(fed)]) == tuple(fed) and (best is None or len(fed) > len(best[1])):
                    best = (name, fed, st)
            if best and (len(ids) - len(best[1])) * TOK_DECODE_S < cost_scratch(len(ids)):
                if best[2] is not None:
                    snap_restore(model, best[2])
                start, src = len(best[1]), f"{best[0]} {len(best[1])} ток + хвост {len(ids) - len(best[1])}"
        LIVE["fed"] = None                                              # пока идём — состояние не определено
        try:
            if start == 0:
                k = min(len(ids), PREFILL_MAX)                       # длиннее предела: начало пачкой, остаток по одному
                out, _, _ = model.forward(torch.tensor([ids[:k]], device="cuda"), 0)
                if k < len(ids):
                    src = f"с нуля: {k} пачкой + {len(ids) - k} по одному"
                start = k
            for i in range(start, len(ids)):
                out, _, _ = model.forward(torch.tensor([[ids[i]]], device="cuda"), i)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"промт {len(ids)} ток: нехватка видеопамяти ({src})")
            on_text("[DeepSeek: не хватило видеопамяти на чтение промта — сократите сообщение или начните новый чат]")
            return 0, len(ids), 0.0, 0.0
        torch.cuda.synchronize(); t_pre = time.time() - t0
        log(f"промт {len(ids)} ток: {src}, {t_pre:.1f} с")
        g.append(int(out[0]))
        pos, t1 = len(ids), time.time()
        while True:
            if g[-1] == tok.eos_token_id:
                g.pop(); break
            text = tok.decode(g, skip_special_tokens=True)
            if not text.endswith("�") and len(text) > len(shown):   # не резать многобайтный символ пополам
                on_text(text[len(shown):]); shown = text
            if len(g) >= max_tokens or cancelled():
                break
            out, _, _ = model.forward(torch.tensor([[g[-1]]], device="cuda"), pos)
            pos += 1
            g.append(int(out[0]))
        text = tok.decode(g, skip_special_tokens=True)
        if len(text) > len(shown):
            on_text(text[len(shown):])
        if PREFIX:
            seq = ids + g + [tok.eos_token_id]                           # в буферах — ровно первые pos токенов этой цепочки
            LIVE["fed"] = tuple(seq[:pos])
            t2 = time.time()
            try:
                snap_take(model, LIVE["fed"])
            except Exception as e:
                log("снимок не сохранён:", e)
            log(f"снимок {pos} ток за {time.time()-t2:.2f} с")
    return len(g), len(ids), t_pre, time.time() - t1


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            pass

    def log_message(self, fmt, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/health"):
            return self._json(200 if STATE["status"] == "ready" else 503,
                              {**STATE, "uptime_s": round(time.time() - STATE["since"]), "busy": GEN_LOCK.locked()})
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "gpu104", "created": 0}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        waited = 0
        while STATE["status"] == "loading" and waited < 600:
            time.sleep(1); waited += 1
        if STATE["status"] != "ready":
            return self._json(503, {"error": {"message": f"модель не готова: {STATE['status']} {STATE['error'] or ''}"}})
        msgs = [conv_msg(m) for m in req.get("messages", [])]
        STATE["nocache"] = bool(req.get("ds_nocache"))          # проверка: читать с нуля, мимо снимков
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or DEFAULT_MAX_TOKENS)
        cid, created, state = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time()), {"broken": False}

        with GEN_LOCK:
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(obj):
                    data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)) + "\n\n").encode()
                    try:
                        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        state["broken"] = True          # «стоп» в чате ⇒ остановить генерацию

                def emit(t):
                    send({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                          "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})

                n, plen, t_pre, dt = generate(msgs, max_tokens, emit, lambda: state["broken"])
                send({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "length" if n >= max_tokens else "stop"}],
                      "usage": {"prompt_tokens": plen, "completion_tokens": n, "total_tokens": plen + n}})
                send("[DONE]")
                try:
                    self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
                except Exception:
                    pass
            else:
                parts = []
                n, plen, t_pre, dt = generate(msgs, max_tokens, parts.append, lambda: False)
                self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                                              "finish_reason": "length" if n >= max_tokens else "stop"}],
                                 "usage": {"prompt_tokens": plen, "completion_tokens": n, "total_tokens": plen + n}})
        STATE["served"] += 1
        log(f"промт {plen} ток за {t_pre:.1f} с, ответ {n} ток за {dt:.1f} с = {max(n - 1, 0) / max(dt, 1e-6):.2f} ток/с"
            + (" (прерван)" if state["broken"] else ""))


if __name__ == "__main__":
    os.chdir(HERE)
    threading.Thread(target=load_engine, daemon=True).start()
    log(f"deepseek-stream на :{PORT}, грузится модель…")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
