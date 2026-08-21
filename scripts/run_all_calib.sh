#!/bin/bash
# 顺序跑黑盒 logp-协方差标定(观测 token 集口径) + 一个全 Vsh 对照
#
# 复现: 在装有 GPU + 依赖(repo 下的 scripts/)的机器上直接运行本脚本。
# 路径可通过环境变量覆盖(默认值对应 autodl):
#   WORK_DIR   代码所在目录(需含 calibrate_logp_cov.py), 默认本脚本所在目录
#   PY         Python 解释器, 默认 /root/miniconda3/bin/python
#   HF_ENDPOINT HuggingFace 镜像, 默认 https://hf-mirror.com
# 结果统一写入 ../data/ (根目录 data/)。

# 本脚本所在目录即代码目录
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORK_DIR:-$SELF_DIR}"
PY="${PY:-/root/miniconda3/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OUT_DIR="${OUT_DIR:-$(cd "$SELF_DIR/.." && pwd)/data}"
mkdir -p "$OUT_DIR"

run() {
  local name="$1"; shift
  echo "=== $(date +%H:%M:%S) START $name ==="
  "$PY" calibrate_logp_cov.py "$@" --out "$OUT_DIR/scan_${name}_logp.txt" > "$OUT_DIR/scan_${name}_logp.log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE  $name (exit $?) ==="
}

run qwen_l8  --model Qwen/Qwen3.5-0.8B      --layer 8
run qwen_l16 --model Qwen/Qwen3.5-0.8B      --layer 16
run llama_l8 --model unsloth/Llama-3.2-1B   --layer 8
run gemma_l8 --model unsloth/gemma-3-1b-pt  --layer 8
run qwen_l8_fullvsh --model Qwen/Qwen3.5-0.8B --layer 8 --full-vsh

echo "=== ALL DONE $(date +%H:%M:%S) ==="
for f in "$OUT_DIR"/scan_*_logp.txt; do echo "--- $f ---"; cat "$f"; done
