#!/usr/bin/env bash
# K2 Horizon 7B sunucusu (systemd: k2-llm.service). Backend: $DATA/backend dosyası (llamacpp|vllm)
# İlk açılışta 12GB'a sığan en büyük context'i dener ve $DATA/llm_state.env'e yazar.
# Yeniden denemek için:  rm ~/.local/share/k2loop/llm_state.env && systemctl --user restart k2-llm
set -euo pipefail
DATA="${K2LOOP_DATA:-$HOME/.local/share/k2loop}"
BACKEND="$(cat "$DATA/backend" 2>/dev/null || echo llamacpp)"
STATE="$DATA/llm_state.env"
PORT="${K2LOOP_PORT:-8080}"
CANDIDATES="${K2LOOP_CTX_CANDIDATES:-65536 49152 40960 32768 24576}"

if [ "$BACKEND" = "vllm" ]; then
  CTX=32768
  # shellcheck source=/dev/null
  [ -f "$STATE" ] && . "$STATE"
  echo "CTX=$CTX" > "$STATE"
  exec "$DATA/vllm-venv/bin/vllm" serve IFM/K2-Horizon-7B --trust-remote-code --dtype bfloat16 \
    --quantization bitsandbytes --reasoning-parser k2_horizon \
    --max-model-len "$CTX" --gpu-memory-utilization 0.90 --max-num-seqs 2 \
    --served-model-name k2-horizon-7b --host 127.0.0.1 --port "$PORT"
fi

BIN="$DATA/llama.cpp/build/bin/llama-server"
MODEL="$(ls "$DATA"/models/*.gguf 2>/dev/null | head -1 || true)"
[ -x "$BIN" ] || { echo "llama-server yok: $BIN (install.sh'i tekrar çalıştırın)"; exit 1; }
[ -n "$MODEL" ] || { echo "GGUF model yok: $DATA/models"; exit 1; }
if "$BIN" --help 2>&1 | grep -- '--flash-attn' | grep -q 'on|off'; then FA=(--flash-attn on); else FA=(--flash-attn); fi
ARGS=(-m "$MODEL" -ngl 999 -np 1 --jinja --host 127.0.0.1 --alias k2-horizon-7b
      --cache-type-k q8_0 --cache-type-v q8_0 "${FA[@]}")

probe() {  # $1 = ctx ; 0 if the server starts and answers /health
  "$BIN" "${ARGS[@]}" -c "$1" --port 18080 >"$DATA/probe.log" 2>&1 &
  local pid=$! t=0
  while [ $t -lt 300 ]; do
    if curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1; then
      # WSL: the Windows driver spills overflowing VRAM into system RAM instead of failing -> "fits" but crawls.
      # Treat a (nearly) full card as "does not fit".
      local used total
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
      total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 99999)"
      kill $pid; wait $pid 2>/dev/null || true
      [ "${used:-0}" -lt $(( ${total:-99999} - 300 )) ] && return 0
      echo "  VRAM doldu (${used}/${total} MiB): bu context sığmıyor"; return 1
    fi
    kill -0 $pid 2>/dev/null || return 1
    sleep 2; t=$((t+2))
  done
  kill $pid 2>/dev/null || true; return 1
}

if [ ! -f "$STATE" ] || [ "${1:-}" = "--reprobe" ]; then
  for c in $CANDIDATES; do
    echo "context deneniyor: $c"
    if probe "$c"; then echo "CTX=$c" > "$STATE"; echo "seçildi: $c"; break; fi
  done
  [ -f "$STATE" ] || { echo "Hiçbir context sığmadı. $DATA/probe.log'a bakın."; exit 1; }
fi
# shellcheck source=/dev/null
. "$STATE"
exec "$BIN" "${ARGS[@]}" -c "$CTX" --port "$PORT"
