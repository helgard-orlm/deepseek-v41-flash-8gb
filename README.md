# DeepSeek-V4.1-Flash (510 GB) on an 8 GB GPU — experts and Engram streamed from NVMe

An engine that runs `deepseek-ai/DeepSeek-V4.1-Flash` (552B backbone + 196B Engram, 384 experts top-6, FP8 + FP4)
on a single **RTX 5060 8 GB** at **~1.6 tokens/s from disk only** and **~2.4 tokens/s with a 16 GB RAM cache**.
Only the dense part lives on the GPU; experts and Engram rows are read on demand from the original
Hugging Face files on an NVMe drive. **No conversion, no repacking:** `setup.sh` changes one line of the
reference code and that is all.

The arithmetic is DeepSeek's own reference code (`inference/model.py` + `kernel.py`, MIT), used as is.
The engine only replaces *where the weights are stored*. Every speed-up below was accepted only if the
generated tokens stayed identical to the plain version: **136/136 tokens** on the 4-question check.

Batch size 1, text only (vision and MTP are not wired in). Code comments are in Russian.

Write-up with the story and measurements: [dev.to post](https://dev.to/helgard_orlm/running-the-510-gb-deepseek-v41-flash-on-an-8-gb-gpu-and-three-bugs-that-never-raise-an-error-5647).

## Hardware it was built on

| part | value |
|---|---|
| GPU | RTX 5060, 8 GB (Blackwell, sm_120), PCIe 5.0 ×8 |
| CPU | Intel Core Ultra 5 225F |
| RAM | 31 GiB |
| model disk | PCIe Gen5 NVMe, ×4 from the CPU, model files only — measured 8.2 GiB/s on expert-sized reads |

## Where the 475 GiB go

| part | size | where it lives |
|---|---|---|
| attention (FP8), shared experts, norms, router | ~6.7 GiB | GPU (cap `--vram_gb 7.3`) |
| `wo_a` | 1.25 GiB | GPU in FP8, expanded to BF16 right before the matmul with the exact `convert.py` formula |
| embeddings + lm_head | | RAM (head in FP32 on the CPU, as in the reference) |
| routed experts: 15,360 × 17.93 MiB (FP4) | 269 GiB | NVMe → RAM LRU cache → GPU slots |
| Engram tables (FP8) | 189 GiB | NVMe, ~48 rows per token read by hash, 5 ms/token |

Per token: 6 experts × 40 layers = **240 experts = 4.2 GiB**. That is ~4× more than Qwen3.8-Flash-Next
(see [qwen-flash-next-8gb](https://github.com/helgard-orlm/qwen-flash-next-8gb)), which is why this model is
slower with the same method.

## The method

1. **Read experts straight from the HF safetensors.** In the original files an expert's weights `w1|w2|w3`
   are already contiguous (17.7 MB), the scales are another contiguous block (1.1 MB) ⇒ two `pread` with
   `O_DIRECT` per expert into page-aligned pinned memory. A read benchmark (`tools/readbench.py`) shows this
   already uses 94–97% of the disk: splitting or re-laying-out the data gives nothing.
   Tensors are read by offset from the safetensors header — `safe_open` maps the whole file, and mapping a
   101 GB Engram shard on a 31 GiB machine fails with `ENOMEM`.
2. **RAM LRU cache** of experts in pinned memory (`--store mixed --ram_gb N`). A simulation over recorded
   routes compared LRU, LFU with ageing, per-layer LRU and SLRU: plain LRU was the best of the simple ones
   (Belady's optimum would be ~15 points higher).
3. **Disk ↔ GPU overlap, "window 2 × 4 pieces"** (scheme by Codex, implemented by Claude): a separate copy
   thread with CUDA events instead of `synchronize`; experts are read **one after another**, each in 4 parallel
   pieces, at most 2 in flight. Issuing all 6 experts at once is *slower*: they share the disk and all finish at
   the end of the layer, so computing cannot start earlier. Disk wait 0.65 → 0.31 s/token.
4. **Short path for one token** (`--fast1 1`): the reference does `torch.where(indices == e)` for every expert,
   which is a CPU↔GPU sync 240 times per token and also delays the next copy. For M = 1 the indices are moved
   to the CPU once per layer — same arithmetic, +5.7%.
5. **Prefetch 2 experts of the next layer into the disk gap** (`--pf 2`): the router of layer i+1 is applied to
   the FFN input of layer i (rescaled by the ratio of the two `ffn_norm` weights); after the required reads of
   layer i are in flight, the top-2 guesses not yet in RAM are read. 94% of them are used. P = 3 and 4 were
   measured and are worse: guesses start to compete with required reads.
   Lesson: "bytes read in vain" is the wrong metric — reads in the gap are free; what counts is how many
   experts arrive on time.
6. **Server** (`ds41_server.py`): OpenAI-compatible `/v1/chat/completions` with streaming and cancel.
   Dialogue snapshots: the full model state (KV, sliding window ring, compression tails, indexer cache,
   Engram token history) is 30.1 MiB; after every answer it is saved to RAM (8 LRU) and to `~/.cache/ds41_snaps`,
   so a follow-up message only processes the new tail. The snapshot key includes a hash of the engine and
   reference code, so stale snapshots are never reused after a code change.
7. **Long prompts in 8 GB:** hyper-connection merges in pieces of 128 positions, sparse attention in pieces of
   256 positions written straight into the output (no `torch.cat`). A prompt of up to ~780 tokens goes in one
   batch; the server uses 700 (`DS_PREFILL_MAX`) and feeds the rest token by token.

## Pitfalls on sm_120 (RTX 50xx) — all three are silent

1. **TileLang 0.1.8** (the version pinned by the reference) gives garbage on sm_120 without any error:
   `fp4_gemm` cosine 0.0006 vs a torch reference, `fp8_gemm` NaN on 17 tokens, the model writes "athaatha Stone".
   The reference self-test passes because it checks shapes, not numbers. **TileLang 0.1.9**: cosine 0.9996.
2. **Race in `act_quant`** of the reference `kernel.py`: with `round_scale` (always on, `scale_fmt = ue8m0`)
   the kernel is built with `num_stages = 0`, and on the RTX 5060 for M > 64 rows part of the FP8 output is NaN,
   **non-deterministically** (the same input five times: 367…2065 NaN). Generation (M = 1) is clean, so only
   prompts are hit. Not the compiler — CUDA 12.8 does the same. `num_stages = 2` gives 0 NaN and is bit-exact
   with a torch implementation for M = 1/17/300/1024. This is the one line `setup.sh` changes.
3. **`sparse_attn` asks for 141 KB of shared memory** (64 heads × 512 per block); consumer Blackwell has ~99 KB.
   The same kernel is called in groups of 16 heads (`--attn_heads`, heads are independent). Note: `.contiguous()`
   of a head slice at 1 token does not change strides, and the kernel checks strides — use
   `clone(memory_format=torch.contiguous_format)`.

## Results

Tokens/s of the answer, single user, average of the long answers of the 4-question check
(all runs: tokens identical to the reference path, 136/136).

| version | disk only | RAM cache 14 GB | prompt 311 tokens |
|---|---|---|---|
| v1: plain streaming | 0.89 | 1.17 | 27 s |
| v2: overlap, window 2 × 4 | 1.27–1.30 | 1.42–1.67 | 17.9 s |
| v3: + short path for one token | 1.35 | 1.95–1.98 | 17.9 s |
| **v4: + prefetch P = 2** | **1.64** | **2.28–2.34** | 17.9 s |
| v4, RAM 18 GB | | 2.37–2.41 | |
| live chat (server, RAM 16 GB) | | **2.43** | |

Peak VRAM 7.06 GiB.

## Tried and rejected (measured)

- **Deltas between experts / layers** for better compression: entropy of FP4 codes 3.895 bits; the delta to the
  same expert in the next layer or to a neighbour is 3.997 bits — worse. Even after optimal neuron permutation and
  per-neuron scaling the relative delta is 0.999–1.000, the same as two random matrices.
- **Skipping "silent" neurons** (not reading their `w2` columns): to keep each expert within 5% error you need
  61–75% of the neurons ⇒ only ~11–13% fewer bytes, and `w2` would have to be read after `w1/w3` are computed.
  SwiGLU neurons here do not go silent (unlike ReLU² models).
- **Predicting the experts 1–3 layers ahead** without training: top-6 hit rate 71% / 65% / 59% — good enough for
  2 guesses into the gap (item 5), not for replacing the real reads.
- Re-laying-out experts on disk, more reader threads, LFU / SLRU cache policies — no gain (see items 1–2).

## Running

```bash
pip install -r requirements.txt            # torch 2.11.0+cu128 for Blackwell; see versions below
export DS_DIR=/path/on/nvme/DeepSeek-V4.1-Flash
./setup.sh      # downloads the snapshot (510 GB) if DS_DIR is empty, checks kernel.py, builds inference_fix/
export LD_LIBRARY_PATH=$(python -c 'import z3, os; print(os.path.join(os.path.dirname(z3.__file__), "lib"))')
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/usr/local/cuda
DS_RAM_GB=16 python ds41_server.py          # :9804, the model loads in ~20 s
```

Then point any OpenAI-compatible client (we use Open WebUI) at `http://<host>:9804/v1`, model `deepseek-v4.1-flash`.
The server unloads Ollama models from the GPU on start (`127.0.0.1:11434`), because they share the card.
Answers are greedy; `max_tokens` defaults to 2048.

The engine alone runs the 4-question check + a 300-token prompt and writes a JSON with tokens and timings:

```bash
python v41_stream_v4.py --dir $DS_DIR --code inference_fix --store mixed --ram_gb 14 --fast1 1 --pf 2
python v41_stream_v4.py --dir $DS_DIR --code inference_fix --store disk --fast1 1 --pf 2
```

| env | meaning | default |
|---|---|---|
| `DS_DIR` | original HF snapshot on NVMe | `/fast/DeepSeek-V4.1-Flash` |
| `DS_RAM_GB` | expert RAM cache | 12 (we use 16) |
| `DS_PORT` | port | 9804 |
| `DS_MAX_SEQ` | context limit | 8192 |
| `DS_PREFILL_MAX` | prompt tokens in one batch, the rest token by token | 700 |
| `DS_SNAP_DIR` | snapshot copies on disk, `""` = RAM only | `~/.cache/ds41_snaps` |

## Notes

- `v41_stream_v4.py` and `ds41_server.py` are byte-identical to the running service.
  `tools/readbench.py` only got `DS_DIR` instead of a hard-coded path.
- Runtime we use: Python 3.13, torch 2.11.0+cu128, tilelang 0.1.9, apache-tvm-ffi 0.1.9, transformers 5.18,
  CUDA toolkit 13.2. Built against the HF revision `dba1be0a40aa45a94ad051997016db3960a90277`.
- Model weights and DeepSeek's code are not included; they are MIT-licensed on the model card.

## Credits

Direction and decisions, the idea of predicting the next layer's experts: helgard. Engine, server, the
`act_quant` race and the other sm_120 fixes: Claude (Anthropic). Overlap scheme (separate copy thread + events),
the one-token short path and the cache-policy question: Codex (OpenAI). Reference implementation: DeepSeek.

License: Apache-2.0 (code in this repository).
