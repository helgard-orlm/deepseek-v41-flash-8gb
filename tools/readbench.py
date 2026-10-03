# Тест чтения экспертов DeepSeek-V4.1-Flash с /fast: как движок (2 чтения O_DIRECT), одно сплошное, дробление на N частей.
# Пачки по 6 экспертов одного слоя (как в генерации), 40 пачек = «токен». Случайные эксперты, без повторов.
import json, os, struct, random, time, mmap, concurrent.futures as cf, sys
D = os.environ.get("DS_DIR", "/fast/DeepSeek-V4.1-Flash"); BLK = 4096
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
meta = {}
for fn in sorted(set(idx.values())):
    p = f"{D}/{fn}"
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    for k, v in h.items():
        if k != "__metadata__": meta[k] = (p, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
fds = {}
def loc(li, e):
    p = f"layers.{li}.ffn.experts.{e}."
    w1 = meta[p + "w1.weight"]; s1 = meta[p + "w1.scale"]
    if w1[0] not in fds: fds[w1[0]] = os.open(w1[0], os.O_RDONLY | os.O_DIRECT)
    WB = sum(meta[p + f"{w}.weight"][2] for w in ("w1", "w2", "w3")); SB = sum(meta[p + f"{w}.scale"][2] for w in ("w1", "w2", "w3"))
    return fds[w1[0]], w1[1], WB, s1[1], SB
NS = 128; SLOT = 20 * 2**20
BUF = mmap.mmap(-1, NS * SLOT)
def segs(off, ln):
    a = off // BLK * BLK; return a, (off - a + ln + BLK - 1) // BLK * BLK
GAP = []
def jobs_for(mode, L):
    fd, wo, WB, so, SB = L; out = []
    wa, wl = segs(wo, WB); sa, sl = segs(so, SB)
    GAP.append(sa - (wa + wl))
    n = 1 if mode == "cur" else int(mode[1:])
    step = (wl // n + BLK - 1) // BLK * BLK
    for i in range(0, wl, step): out.append((fd, wa + i, min(step, wl - i)))
    out.append((fd, sa, sl)); return out
pool = cf.ThreadPoolExecutor(64)
def rd(j):
    (fd, off, ln), k = j
    mv = memoryview(BUF)[k * SLOT: k * SLOT + ln]
    return os.preadv(fd, [mv], off)
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
used = set()
def pick(li):
    while True:
        e = random.randrange(384)
        if (li, e) not in used: used.add((li, e)); return e
for mode in ["cur", "s2", "s4", "s8", "s16", "cur", "s4"]:
    tot, t0 = 0, time.time()
    for li in range(40):                       # один «токен»: 40 слоёв × 6 экспертов, слой ждёт свои 6
        js = [j for e in [pick(li) for _ in range(6)] for j in jobs_for(mode, loc(li, e))]
        tot += sum(pool.map(rd, [(j, i) for i, j in enumerate(js)]))
    dt = time.time() - t0
    print(f"{mode:4s} {tot/2**20:7.0f} МиБ за {dt:.3f} с = {tot/dt/2**30:.2f} ГиБ/с  чтений/слой {len(js)}", flush=True)
print("зазор веса→масштабы, МиБ: мин", min(GAP) / 2**20, "макс", max(GAP) / 2**20)
