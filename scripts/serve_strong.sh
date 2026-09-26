#!/usr/bin/env bash
# Qwen3.8 27B Q4_K_M — llama-swap starts it on demand (after unloading K2 from the GPU).
# The model does not fit in 12 GB VRAM: llama.cpp's own '--fit on' decides how many layers go to the GPU from
# the REAL free VRAM. The old "does it start with N layers?" probe was fooled on WSL: the Windows NVIDIA driver
# spills overflowing VRAM into system RAM (Sysmem Fallback) instead of failing, so ngl=99 "worked" and Qwen ran
# very slowly over PCIe.
# Env: K2LOOP_STRONG_CTX (default 32768), K2LOOP_FIT_MARGIN in MiB (default 1024),
#      K2LOOP_STRONG_THREADS (default: half the hardware threads = physical cores; SMT threads slow it down)
set -euo pipefail
DATA="${K2LOOP_DATA:-$HOME/.local/share/k2loop}"
BIN="$DATA/llama.cpp-main/build/bin/llama-server"
MODEL="$(ls "$DATA"/models-strong/*.gguf 2>/dev/null | head -1 || true)"
PORT="${K2LOOP_STRONG_PORT:-8081}"
CTX="${K2LOOP_STRONG_CTX:-32768}"
[ -x "$BIN" ] || { echo "llama-server (ana dal) yok: $BIN"; exit 1; }
[ -n "$MODEL" ] || { echo "Qwen modeli yok: $DATA/models-strong"; exit 1; }
if "$BIN" --help 2>&1 | grep -- '--flash-attn' | grep -q 'on|off'; then FA=(--flash-attn on); else FA=(--flash-attn); fi
printf "CTX=%s\nFIT=on\n" "$CTX" > "$DATA/strong_state.env"   # k2loop's planner reads the context size here
echo "Qwen3.8 27B: ctx=$CTX, GPU katmanları --fit ile (pay ${K2LOOP_FIT_MARGIN:-1024} MiB)"
exec "$BIN" -m "$MODEL" -np 1 --jinja --host 127.0.0.1 --alias qwen3.8-27b -t "${K2LOOP_STRONG_THREADS:-$(( $(nproc) / 2 ))}" \
  --cache-type-k q8_0 --cache-type-v q8_0 "${FA[@]}" \
  -c "$CTX" --fit on --fit-target "${K2LOOP_FIT_MARGIN:-1024}" --port "$PORT"
