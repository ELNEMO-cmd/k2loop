#!/usr/bin/env bash
# Yardımcı model — CPU'da (GPU'ya dokunmaz). llama-swap başlatır, K2 ile AYNI ANDA açık kalır.
# İşleri: niyet sınıflandırma, ToT hipotez puanlama, araştırma özetleri (Local Deep Research), Open WebUI başlık/etiket.
# Ayarlar (env): K2LOOP_HELPER_THREADS (varsayılan: çekirdeklerin yarısı), K2LOOP_HELPER_CTX (varsayılan 32768)
set -euo pipefail
DATA="${K2LOOP_DATA:-$HOME/.local/share/k2loop}"
BIN="$DATA/llama.cpp-main/build/bin/llama-server"
MODEL="$(ls "$DATA"/models-helper/*.gguf 2>/dev/null | head -1 || true)"
PORT="${K2LOOP_PORT:-8082}"
THREADS="${K2LOOP_HELPER_THREADS:-$(( $(nproc) / 2 ))}"
[ -x "$BIN" ] || { echo "llama-server (ana dal) yok: $BIN (install.sh'i tekrar çalıştırın)"; exit 1; }
[ -n "$MODEL" ] || { echo "Yardımcı model yok: $DATA/models-helper"; exit 1; }
export CUDA_VISIBLE_DEVICES=""   # saf CPU: 3080 Ti tamamen K2/Qwen'e kalsın
exec "$BIN" -m "$MODEL" -ngl 0 -t "$THREADS" -c "${K2LOOP_HELPER_CTX:-32768}" -np 2 --jinja \
  --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" --alias helper
