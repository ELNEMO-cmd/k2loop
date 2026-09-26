#!/usr/bin/env bash
# Qwen3.6-35B-A3B (MoE, 3B active) — the "strong" model; llama-swap starts it after unloading K2.
# 35B does not fit in 12 GB, but only ~3B parameters are active per token: attention/shared weights stay on the
# GPU, expert weights live in RAM and run on the CPU. Reported ~20-25 tok/s on an RTX 3080 Ti.
# Env: K2LOOP_MOE_CTX (default 32768)
#      K2LOOP_MOE_NCPU=N  -> all layers on GPU except the experts of the first N layers (manual tuning);
#                            unset -> llama.cpp '--fit on' decides from the real free VRAM
#      K2LOOP_MOE_THREADS (default: physical cores), K2LOOP_FIT_MARGIN MiB (default 1024)
set -euo pipefail
DATA="${K2LOOP_DATA:-$HOME/.local/share/k2loop}"
BIN="$DATA/llama.cpp-main/build/bin/llama-server"
MODEL="$(find "$DATA/models-moe" -name '*.gguf' 2>/dev/null | sort | head -1)"
PORT="${K2LOOP_PORT:-8083}"
CTX="${K2LOOP_MOE_CTX:-32768}"
[ -x "$BIN" ] || { echo "llama-server (ana dal) yok: $BIN"; exit 1; }
[ -n "$MODEL" ] || { echo "MoE modeli yok: $DATA/models-moe"; exit 1; }
if [ -n "${K2LOOP_MOE_NCPU:-}" ]; then
  PLACE=(-ngl all --n-cpu-moe "$K2LOOP_MOE_NCPU" --fit off)
else
  PLACE=(--fit on --fit-target "${K2LOOP_FIT_MARGIN:-1024}")
fi
printf "CTX=%s\n" "$CTX" > "$DATA/strong_state.env"   # k2loop's planner reads the context size here
echo "Qwen3.6-35B-A3B: ctx=$CTX yerlesim: ${PLACE[*]}"
exec "$BIN" -m "$MODEL" -np 1 --jinja --host 127.0.0.1 --alias qwen3.6-35b-a3b \
  -t "${K2LOOP_MOE_THREADS:-$(( $(nproc) / 2 ))}" --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  -c "$CTX" "${PLACE[@]}" --port "$PORT"
