#!/usr/bin/env bash
# Prepare DeepSeek-V4.1-Flash for ds41_server.py.
# Nothing is repacked: the engine reads experts and Engram rows straight from the original HF files.
# This script only (1) downloads the snapshot if it is missing, (2) builds inference_fix/ =
# the reference inference/ code with ONE line changed in kernel.py, (3) links encoding/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DS_DIR=${DS_DIR:?set DS_DIR=/path/on/nvme/DeepSeek-V4.1-Flash}
REPO=deepseek-ai/DeepSeek-V4.1-Flash
REV=dba1be0a40aa45a94ad051997016db3960a90277      # revision the engine was built and checked against
KERNEL_ORIG=1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455
KERNEL_FIXED=0b47cf35aa1173998e24b03fc148a33fc1a7b71786fcbbd63853be697046072d
NEED_GB=520

if [ ! -f "$DS_DIR/model.safetensors.index.json" ]; then
  mkdir -p "$DS_DIR"
  free=$(df -B1G --output=avail "$DS_DIR" | tail -1 | tr -d ' ')
  if [ "$free" -lt "$NEED_GB" ]; then
    echo "need $NEED_GB GB free in $DS_DIR, have $free GB" >&2; exit 1
  fi
  echo "downloading $REPO@$REV (510 GB) into $DS_DIR"
  hf download "$REPO" --revision "$REV" --local-dir "$DS_DIR"
fi

k="$DS_DIR/inference/kernel.py"
got=$(sha256sum "$k" | cut -d' ' -f1)
if [ "$got" != "$KERNEL_ORIG" ]; then
  echo "unexpected $k (sha256 $got) - other revision? the fix below targets $REV" >&2; exit 1
fi

# inference_fix/ holds only symlinks + one patched file
mkdir -p "$HERE/inference_fix"
find "$HERE/inference_fix" -mindepth 1 -maxdepth 1 \( -type l -o -name kernel.py \) -exec rm -f {} +
for f in "$DS_DIR"/inference/*; do
  [ "$(basename "$f")" = kernel.py ] || ln -s "$f" "$HERE/inference_fix/"
done
sed 's/^    num_stages = 0 if round_scale or inplace else 2$/    num_stages = 2  # fix: num_stages=0 races on sm_120 (RTX 5060), NaN in FP8 output when M>64/' \
  "$k" > "$HERE/inference_fix/kernel.py"
got=$(sha256sum "$HERE/inference_fix/kernel.py" | cut -d' ' -f1)
[ "$got" = "$KERNEL_FIXED" ] || { echo "kernel.py patch did not apply (sha256 $got)" >&2; exit 1; }
diff "$k" "$HERE/inference_fix/kernel.py" || true

ln -sfn "$DS_DIR/encoding" "$HERE/encoding"
echo "ok: inference_fix/ and encoding/ ready; start with  DS_DIR=$DS_DIR DS_RAM_GB=16 python ds41_server.py"
