#!/usr/bin/env bash
# ============================================================================
#  k2loop 0.2 kurulum — Zorin OS 17/18 (Ubuntu 22.04/24.04 tabanlı), TAMAMEN YEREL
#
#  GPU (3080 Ti): K2 Horizon 7B (kod) ⇄ Qwen3.8 27B (plan/eskalasyon/review)   — llama-swap yönetir
#  CPU          : yardımcı model (Qwen3 4B), Laya, testler, Local Deep Research, SearXNG
#  Arayüzler    : Open WebUI :3000 (canlı ajan) · Phoenix :6006 (ToT iz ağacı)
#                 llama-swap :8080/ui (modeller) · Local Deep Research :5000 (araştırma)
#
#  Kullanım:   ./install.sh               (önerilen)
#              ./install.sh --no-local    (Qwen3.8 27B'yi kurma)
#              ./install.sh --no-web      (SearXNG / LDR / Open WebUI kurma)
#              ./install.sh --hybrid      (Claude Code'u da kur ve hibrit moda geç)
#              ./install.sh --quant Q5_K_M
#  İki parçalı (sudo şifresi sormadan, ör. WSL'de Windows tarafından):
#              sudo ./install.sh --root-part --for-user KULLANICI   (apt, docker, CUDA, linger)
#              ./install.sh --skip-root                            (geri kalan her şey, sudo kullanmaz)
#  Tekrar çalıştırmak güvenlidir: yapılmış adımları atlar.
# ============================================================================
set -Eeuo pipefail
# systemd-run / cron gibi ortamlarda HOME ve USER boş olabilir
HOME="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"; USER="${USER:-$(id -un)}"; export HOME USER
export DEBIAN_FRONTEND=noninteractive
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${K2LOOP_DATA:-$HOME/.local/share/k2loop}"
CONF="$HOME/.config/k2loop"
UNITS="$HOME/.config/systemd/user"
BACKEND="llamacpp"; QUANT="Q4_K_M"; SKIP_LLM=0; SKIP_LAYA=0; WITH_LOCAL=1; WITH_WEB=1; HYBRID=0
ROOT_PART=0; SKIP_ROOT=0; FOR_USER=""
STRONG_REPO="${K2LOOP_STRONG_REPO:-bartowski/Qwen3.8-27B-GGUF}"; STRONG_FILE="${K2LOOP_STRONG_FILE:-Qwen3.8-27B-Q4_K_M.gguf}"
STRONG_KIND="${K2LOOP_STRONG_KIND:-moe}"   # moe = Qwen3.6-35B-A3B (~21 tok/s on a 3080 Ti) | dense = Qwen3.8 27B (~4 tok/s)
MOE_REPO="${K2LOOP_MOE_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"; MOE_QUANT="${K2LOOP_MOE_QUANT:-IQ4_XS}"
HELPER_REPO="${K2LOOP_HELPER_REPO:-unsloth/Qwen3-4B-Instruct-2507-GGUF}"; HELPER_FILE="${K2LOOP_HELPER_FILE:-Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
GGUF_REPO="${K2LOOP_GGUF_REPO:-AMAImedia/K2-Horizon-7B-GGUF}"
BERNSTEIN_VERSION="3.20.0"; LAYA_VERSION="0.3.20"
LOG="$HOME/k2loop-install.log"

while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --quant) QUANT="$2"; shift 2;;
    --skip-llm) SKIP_LLM=1; shift;;
    --skip-laya) SKIP_LAYA=1; shift;;
    --local) shift;;                      # eski seçenek: artık varsayılan
    --no-local) WITH_LOCAL=0; shift;;
    --no-web) WITH_WEB=0; shift;;
    --hybrid) HYBRID=1; shift;;
    --strong) STRONG_KIND="$2"; shift 2;;
    --root-part) ROOT_PART=1; shift;;
    --for-user) FOR_USER="$2"; shift 2;;
    --skip-root) SKIP_ROOT=1; shift;;
    -h|--help) sed -n 2,19p "$0"; exit 0;;
    *) echo "Bilinmeyen seçenek: $1"; exit 1;;
  esac
done

exec > >(tee -a "$LOG") 2>&1
say()  { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
ok()   { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m  ! %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31m  ✗ %s\033[0m\n" "$*"; echo "  Ayrıntı: $LOG"; exit 1; }
trap 'die "Beklenmeyen hata (satır $LINENO). Script tekrar çalıştırılabilir."' ERR

# shellcheck source=/dev/null
. /etc/os-release
case " ${ID:-} ${ID_LIKE:-} " in *" ubuntu "*|*" zorin "*|*" debian "*) ;; *) die "Bu script Zorin/Ubuntu için.";; esac
case "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}" in
  noble) UBV=2404;; jammy) UBV=2204;;
  *) die "Desteklenmeyen sürüm (${PRETTY_NAME:-?}). Zorin 17 (jammy) veya 18 (noble) gerekli.";;
esac
IS_WSL=0; grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
# WSL: nvidia-smi ve CUDA kütüphaneleri Windows sürücüsünden /usr/lib/wsl/lib altında gelir
# (etkileşimli kabukta PATH'te olur, systemd servislerinde olmaz)
WSL_LIB=""; [ -d /usr/lib/wsl/lib ] && WSL_LIB="/usr/lib/wsl/lib" && export PATH="$PATH:$WSL_LIB"
ok "Sistem: ${PRETTY_NAME:-$ID} (Ubuntu ${UBV:0:2}.${UBV:2})$([ "$IS_WSL" -eq 1 ] && echo ' — WSL2')"
if [ "$IS_WSL" -eq 1 ]; then
  [ "$(ps -p 1 -o comm=)" = "systemd" ] || die "WSL'de systemd kapalı: /etc/wsl.conf içine [boot] systemd=true yazın, Windows'ta 'wsl --shutdown'."
  WIN_KB="$(df -Pk /mnt/c 2>/dev/null | awk 'NR==2{print $4}' || true)"
  WIN_FREE=$(( ${WIN_KB:-0} / 1024 / 1024 ))
  ok "Windows C: boş alan: ${WIN_FREE} GB (WSL diski oradadır)"
  [ "$WIN_FREE" -lt 55 ] && warn "C: sürücüsünde ~55 GB boş alan önerilir (modeller + CUDA + docker)."
fi

PKGS="git curl wget ca-certificates build-essential cmake pkg-config jq openssl"
[ "$WITH_WEB" -eq 1 ] && PKGS="$PKGS docker.io"
if [ "$EUID" -eq 0 ]; then SUDO=(); else SUDO=(sudo); fi

install_cuda() {  # CUDA derleyicisi (sürücüye dokunmaz); root ya da sudo ile
  command -v nvcc >/dev/null && return 0
  local drv major pkg repo
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"; major="${drv%%.*}"
  if   [ "$major" -ge 570 ]; then pkg=cuda-toolkit-12-8
  elif [ "$major" -ge 560 ]; then pkg=cuda-toolkit-12-6
  elif [ "$major" -ge 550 ]; then pkg=cuda-toolkit-12-4
  else die "Sürücü $drv çok eski (>=550 gerekli). Daha yeni bir NVIDIA sürücüsü kurun."; fi
  # WSL: NVIDIA'nın WSL deposu (Windows sürücüsünün libcuda'sına dokunmayan paketler)
  repo="ubuntu${UBV}"; [ "$IS_WSL" -eq 1 ] && repo="wsl-ubuntu"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
  "${SUDO[@]}" dpkg -i /tmp/cuda-keyring.deb >/dev/null
  "${SUDO[@]}" apt-get update -qq
  "${SUDO[@]}" apt-get install -y -qq "$pkg" >/dev/null
}

check_gpu() {
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    die "NVIDIA sürücüsü çalışmıyor. Zorin: Yazılım ve Güncellemeler → Ek Sürücüler → 'nvidia-driver-5xx
     (proprietary, tested)' (veya: sudo ubuntu-drivers install), yeniden başlatın. WSL: Windows'ta güncel NVIDIA
     sürücüsünü kurun. Sonra bu script'i tekrar çalıştırın."
  fi
}

# ---------------------------------------------------------------- root parçası (--root-part)
if [ "$ROOT_PART" -eq 1 ]; then
  [ "$EUID" -eq 0 ] || die "--root-part root olarak çalışmalı (sudo ./install.sh --root-part --for-user KULLANICI)"
  [ -n "$FOR_USER" ] && id "$FOR_USER" >/dev/null 2>&1 || die "--for-user ile geçerli bir kullanıcı verin"
  say "Root adımları ($FOR_USER için): paketler, docker, CUDA, linger"
  apt-get update -qq
  # shellcheck disable=SC2086
  apt-get install -y -qq $PKGS >/dev/null
  ok "paketler: $PKGS"
  if [ "$WITH_WEB" -eq 1 ]; then
    systemctl enable --now docker >/dev/null 2>&1 || warn "docker servisi açılamadı"
    usermod -aG docker "$FOR_USER"
    ok "docker (kullanıcı '$FOR_USER' docker grubunda; yeni oturumda geçerli)"
  fi
  check_gpu
  [ "$SKIP_LLM" -eq 1 ] || { install_cuda; ok "CUDA: $(/usr/local/cuda/bin/nvcc --version | grep -o 'release [0-9.]*')"; }
  loginctl enable-linger "$FOR_USER" || warn "linger açılamadı"
  ok "Root adımları bitti. Şimdi '$FOR_USER' olarak: ./install.sh --skip-root"
  exit 0
fi
[ "$EUID" -eq 0 ] && die "root olarak çalıştırmayın (root adımları için: --root-part --for-user KULLANICI)."
# systemctl --user için (ör. 'wsl -u KULLANICI' ile açılan oturumlarda boş olabilir)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
RAM_GB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
FREE_GB=$(( $(df -Pk "$HOME" | awk 'NR==2{print $4}') / 1024 / 1024 ))
ok "RAM: ${RAM_GB} GB, boş disk: ${FREE_GB} GB, CPU: $(nproc) iş parçacığı"
[ "$WITH_LOCAL" -eq 1 ] && [ "$RAM_GB" -lt 28 ] && warn "Qwen3.8 27B için ~32 GB RAM önerilir (sizde ${RAM_GB} GB)."
[ "$FREE_GB" -lt 55 ] && warn "En az ~55 GB boş disk önerilir (modeller + CUDA + derlemeler + docker imajları)."
mkdir -p "$DATA/bin" "$DATA/models" "$DATA/models-helper" "$CONF" "$UNITS" "$HOME/.local/bin"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"

# ---------------------------------------------------------------- 1. apt
say "1/10 Sistem paketleri"
if [ "$SKIP_ROOT" -eq 1 ]; then
  for c in git curl wget cmake jq openssl g++; do command -v "$c" >/dev/null || die "$c yok: önce root adımı (--root-part)"; done
  [ "$WITH_WEB" -eq 0 ] || command -v docker >/dev/null || die "docker yok: önce root adımı (--root-part)"
else
  sudo apt-get update -qq
  # shellcheck disable=SC2086
  sudo apt-get install -y -qq $PKGS >/dev/null
  if [ "$WITH_WEB" -eq 1 ]; then sudo systemctl enable --now docker >/dev/null 2>&1 || warn "docker servisi açılamadı"; fi
fi
ok "git, curl, cmake, derleyici$([ "$WITH_WEB" -eq 1 ] && echo ', docker')"

# ---------------------------------------------------------------- 2. GPU
say "2/10 NVIDIA sürücüsü"
check_gpu
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
ok "GPU: $GPU  sürücü: $DRV"

# ---------------------------------------------------------------- 3. python araçları
say "3/10 uv, Bernstein, Aider, k2loop, Phoenix, Local Deep Research"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 >/dev/null
uv tool install -q --python 3.12 --force "bernstein==$BERNSTEIN_VERSION"
uv tool install -q --python 3.12 --force aider-chat
uv tool install -q --python 3.12 --force ruff
uv tool install -q --python 3.12 --force --reinstall \
  --with opentelemetry-sdk --with opentelemetry-exporter-otlp-proto-http "$HERE"
uv tool install -q --python 3.12 --force arize-phoenix || warn "Phoenix kurulamadı (iz ağacı olmadan devam)"
if [ "$WITH_WEB" -eq 1 ]; then
  uv tool install -q --python 3.12 --force local-deep-research || warn "Local Deep Research kurulamadı (SearXNG'ye düşülür)"
fi
uv tool update-shell >/dev/null 2>&1 || true
ok "bernstein $(bernstein --version 2>/dev/null | head -1 || true), aider, ruff, k2loop"
git config --global user.email >/dev/null || warn "git kimliği yok: git config --global user.name 'Ad'; git config --global user.email 'mail'"

# ---------------------------------------------------------------- 4. Claude (sadece --hybrid)
if [ "$HYBRID" -eq 1 ]; then
  say "4/10 Claude Code (hibrit mod)"
  command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash
  command -v claude >/dev/null && ok "claude $(claude --version 2>/dev/null | head -1)" || warn "claude kurulamadı"
else
  say "4/10 Claude atlandı (tam yerel kurulum)"
fi

# ---------------------------------------------------------------- 5. Laya (CPU)
say "5/10 Laya karar modeli (CPU, VRAM kullanmaz)"
if [ "$SKIP_LAYA" -eq 0 ]; then
  [ -x "$DATA/laya-venv/bin/python" ] || uv venv -q --python 3.12 "$DATA/laya-venv"
  uv pip install -q --python "$DATA/laya-venv/bin/python" --index-url https://download.pytorch.org/whl/cpu torch
  uv pip install -q --python "$DATA/laya-venv/bin/python" "laya==$LAYA_VERSION"
  cp "$HERE/src/k2loop/laya_server.py" "$DATA/laya_server.py"
  "$DATA/laya-venv/bin/python" -c "import laya; laya.load('convaiinnovations/laya-typed-decisions', device='cpu'); print('model indirildi')"
  ok "Laya $LAYA_VERSION hazır"
else warn "Laya atlandı (--skip-laya)"; fi

# ---------------------------------------------------------------- 6. llama.cpp derlemeleri + modeller
say "6/10 llama.cpp (K2 fork'u + ana dal) ve modeller"
echo "$BACKEND" > "$DATA/backend"
for s in serve_llm.sh serve_strong.sh serve_moe.sh serve_helper.sh; do cp "$HERE/scripts/$s" "$DATA/bin/$s"; chmod +x "$DATA/bin/$s"; done
build_llama() {  # $1 = kaynak klasörü
  [ -x "$1/build/bin/llama-server" ] && return 0
  echo "  $(basename "$1") derleniyor (5-15 dk)..."
  cmake -S "$1" -B "$1/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$1/build" --config Release -j"$(nproc)" --target llama-server >/dev/null
}
if [ "$SKIP_LLM" -eq 1 ]; then warn "LLM atlandı (--skip-llm)"
else
  if ! command -v nvcc >/dev/null; then
    [ "$SKIP_ROOT" -eq 1 ] && die "nvcc (CUDA) yok: önce root adımı (--root-part)"
    install_cuda
  fi
  ok "nvcc $(nvcc --version | grep -o 'release [0-9.]*')"
  # ana dal: Qwen + yardımcı model
  if [ ! -d "$DATA/llama.cpp-main/.git" ]; then git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$DATA/llama.cpp-main"
  else git -C "$DATA/llama.cpp-main" pull -q --ff-only || true; fi
  build_llama "$DATA/llama.cpp-main"; ok "llama-server (ana dal)"
  if [ "$BACKEND" = "vllm" ]; then
    warn "vLLM backend deneysel."
    [ -x "$DATA/vllm-venv/bin/python" ] || uv venv -q --python 3.12 "$DATA/vllm-venv"
    uv pip install -q --python "$DATA/vllm-venv/bin/python" vllm bitsandbytes
  else
    if [ ! -d "$DATA/llama.cpp/.git" ]; then git clone -q --depth 1 -b model/K2Horizon https://github.com/MBZUAI-IFM/llama.cpp "$DATA/llama.cpp"
    else git -C "$DATA/llama.cpp" pull -q --ff-only || true; fi
    build_llama "$DATA/llama.cpp"; ok "llama-server (K2 fork'u)"
    F="K2-Horizon-7B-${QUANT}.gguf"
    if [ ! -f "$DATA/models/$F" ]; then
      rm -f "$DATA"/models/*.gguf "$DATA/llm_state.env"
      uvx -q --from huggingface_hub hf download "$GGUF_REPO" "$F" --local-dir "$DATA/models"
    fi
    ok "K2: $F"
  fi
  if [ ! -f "$DATA/models-helper/$HELPER_FILE" ]; then
    rm -f "$DATA"/models-helper/*.gguf
    uvx -q --from huggingface_hub hf download "$HELPER_REPO" "$HELPER_FILE" --local-dir "$DATA/models-helper"
  fi
  ok "yardımcı model (CPU): $HELPER_FILE"
  if [ "$WITH_LOCAL" -eq 1 ] && [ "$STRONG_KIND" = "moe" ]; then
    mkdir -p "$DATA/models-moe"
    if ! find "$DATA/models-moe" -name "*${MOE_QUANT}*.gguf" | grep -q .; then
      echo "  indiriliyor: $MOE_REPO ($MOE_QUANT, ~18 GB)"
      uvx -q --from huggingface_hub hf download "$MOE_REPO" --include "*${MOE_QUANT}*" --local-dir "$DATA/models-moe"
    fi
    ok "Qwen3.6-35B-A3B hazır (MoE; uzmanlar CPU'da, gerisi GPU'da)"
  elif [ "$WITH_LOCAL" -eq 1 ]; then
    mkdir -p "$DATA/models-strong"
    if [ ! -f "$DATA/models-strong/$STRONG_FILE" ]; then
      echo "  indiriliyor: $STRONG_REPO / $STRONG_FILE (~16.5 GB)"
      uvx -q --from huggingface_hub hf download "$STRONG_REPO" "$STRONG_FILE" --local-dir "$DATA/models-strong"
      rm -f "$DATA/strong_state.env"
    fi
    ok "Qwen3.8 27B hazır (ihtiyaç anında GPU'ya yüklenir)"
  fi
fi

# ---------------------------------------------------------------- 7. llama-swap
say "7/10 llama-swap (tüm modeller tek adreste: http://127.0.0.1:8080, arayüz /ui)"
if [ ! -x "$DATA/bin/llama-swap" ]; then
  URL="$(curl -fsSL https://api.github.com/repos/mostlygeek/llama-swap/releases/latest \
        | jq -r '.assets[] | select(.name | test("linux_amd64\\.tar\\.gz$")) | .browser_download_url' | head -1)"
  [ -n "$URL" ] || die "llama-swap sürümü bulunamadı (GitHub API). İnternet bağlantısını kontrol edip tekrar deneyin."
  TMP="$(mktemp -d)"; curl -fsSL "$URL" | tar xz -C "$TMP"
  install -m 755 "$(find "$TMP" -type f -name llama-swap | head -1)" "$DATA/bin/llama-swap"; rm -rf "$TMP"
fi
ln -sf "$DATA/bin/llama-swap" "$HOME/.local/bin/llama-swap"
ok "llama-swap $("$DATA/bin/llama-swap" --version 2>/dev/null | head -1 || true)"
# every strong model that is on disk gets an entry, so switching is only 'k2loop strong moe|dense'
QWEN_BLOCK=""; GPU_MEMBERS='"k2-horizon-7b"'
if [ "$WITH_LOCAL" -eq 1 ] && ls "$DATA"/models-strong/*.gguf >/dev/null 2>&1; then
  QWEN_BLOCK="$QWEN_BLOCK
  \"qwen3.8-27b\":
    name: \"Qwen3.8 27B (yoğun; GPU+RAM, yavaş)\"
    cmd: env K2LOOP_DATA=$DATA K2LOOP_STRONG_PORT=\${PORT} $DATA/bin/serve_strong.sh
    proxy: http://127.0.0.1:\${PORT}
    checkEndpoint: /health
    ttl: 900"
  GPU_MEMBERS="$GPU_MEMBERS, \"qwen3.8-27b\""
fi
if [ "$WITH_LOCAL" -eq 1 ] && find "$DATA/models-moe" -name '*.gguf' 2>/dev/null | grep -q .; then
  QWEN_BLOCK="$QWEN_BLOCK
  \"qwen3.6-35b-a3b\":
    name: \"Qwen3.6-35B-A3B (MoE; plan/eskalasyon/review)\"
    cmd: env K2LOOP_DATA=$DATA K2LOOP_PORT=\${PORT} $DATA/bin/serve_moe.sh
    proxy: http://127.0.0.1:\${PORT}
    checkEndpoint: /health
    ttl: 900"
  GPU_MEMBERS="$GPU_MEMBERS, \"qwen3.6-35b-a3b\""
fi
cat > "$CONF/llama-swap.yaml" <<YAML
# install.sh tarafından üretildi. GPU grubunda aynı anda tek model çalışır (K2 ⇄ Qwen),
# CPU grubundaki yardımcı model her zaman açık kalır.
healthCheckTimeout: 3600      # ilk açılışta K2/Qwen için uygun context/katman sayısı ölçülür
logLevel: info
startPort: 10001
models:
  "k2-horizon-7b":
    name: "K2 Horizon 7B (GPU, kod)"
    cmd: env K2LOOP_DATA=$DATA K2LOOP_PORT=\${PORT} $DATA/bin/serve_llm.sh
    proxy: http://127.0.0.1:\${PORT}
    checkEndpoint: /health
    ttl: 0
  "helper":
    name: "Yardımcı model (CPU)"
    cmd: env K2LOOP_DATA=$DATA K2LOOP_PORT=\${PORT} $DATA/bin/serve_helper.sh
    proxy: http://127.0.0.1:\${PORT}
    checkEndpoint: /health
    ttl: 0$QWEN_BLOCK
groups:
  gpu:
    swap: true
    exclusive: false
    members: [$GPU_MEMBERS]
  cpu:
    swap: false
    exclusive: false
    persistent: true
    members: ["helper"]
hooks:
  on_startup:
    preload: ["k2-horizon-7b", "helper"]
YAML
ok "$CONF/llama-swap.yaml"

# ---------------------------------------------------------------- 8. web: SearXNG, Local Deep Research, Open WebUI
SECRETS="$CONF/secrets.env"
touch "$SECRETS"; chmod 600 "$SECRETS"
grep -q '^AGENT_API_KEY=' "$SECRETS" || echo "AGENT_API_KEY=$(openssl rand -hex 16)" >> "$SECRETS"
grep -q 'LDR_USER' "$SECRETS" || printf '%s\n' "# Local Deep Research hesabın (http://127.0.0.1:5000 adresinde kendin oluştur):" \
  "#LDR_USER=" "#LDR_PASSWORD=" >> "$SECRETS"
AGENT_KEY="$(grep '^AGENT_API_KEY=' "$SECRETS" | cut -d= -f2)"
if [ "$WITH_WEB" -eq 1 ]; then
  say "8/10 SearXNG (:8888), Open WebUI (:3000)"
  mkdir -p "$DATA/searxng"
  if [ ! -f "$DATA/searxng/settings.yml" ]; then
    cat > "$DATA/searxng/settings.yml" <<YAML
use_default_settings: true
server:
  secret_key: "$(openssl rand -hex 32)"
  limiter: false
  image_proxy: false
search:
  formats: [html, json]
YAML
  fi
  if docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo docker); fi
  "${DOCKER[@]}" inspect k2-searxng >/dev/null 2>&1 || "${DOCKER[@]}" run -d --name k2-searxng --restart unless-stopped \
    -p 127.0.0.1:8888:8080 -v "$DATA/searxng:/etc/searxng" searxng/searxng >/dev/null
  ok "SearXNG: http://127.0.0.1:8888"
  "${DOCKER[@]}" inspect k2-webui >/dev/null 2>&1 || "${DOCKER[@]}" run -d --name k2-webui --restart unless-stopped \
    --network host -v k2-webui:/app/backend/data \
    -e PORT=3000 -e HOST=127.0.0.1 -e ENABLE_OLLAMA_API=false \
    -e "OPENAI_API_BASE_URLS=http://127.0.0.1:8090/v1;http://127.0.0.1:8080/v1" \
    -e "OPENAI_API_KEYS=$AGENT_KEY;none" \
    -e TASK_MODEL_EXTERNAL=helper -e AIOHTTP_CLIENT_TIMEOUT=86400 \
    ghcr.io/open-webui/open-webui:main >/dev/null
  ok "Open WebUI: http://127.0.0.1:3000"
else say "8/10 web servisleri atlandı (--no-web)"; fi

# ---------------------------------------------------------------- 9. servisler
say "9/10 Arka plan servisleri (systemd --user)"
for old in k2-llm k2-strong; do   # 0.1'in model servisleri: artık llama-swap yönetiyor
  systemctl --user disable --now "$old.service" >/dev/null 2>&1 || true
  rm -f "$UNITS/$old.service"
done
UPATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin${WSL_LIB:+:$WSL_LIB}"
cat > "$UNITS/k2-swap.service" <<UNIT
[Unit]
Description=k2loop: llama-swap (K2 / Qwen / yardımcı model)
After=network.target
[Service]
Environment=K2LOOP_DATA=$DATA
Environment=PATH=$UPATH
ExecStart=$DATA/bin/llama-swap --config $CONF/llama-swap.yaml --listen 127.0.0.1:8080 --watch-config
Restart=on-failure
RestartSec=10
[Install]
WantedBy=default.target
UNIT
cat > "$UNITS/k2-laya.service" <<UNIT
[Unit]
Description=k2loop: Laya decision model (CPU)
[Service]
ExecStart=$DATA/laya-venv/bin/python $DATA/laya_server.py --port 8765 --device cpu
Restart=on-failure
RestartSec=10
[Install]
WantedBy=default.target
UNIT
cat > "$UNITS/k2-agent.service" <<UNIT
[Unit]
Description=k2loop: sürekli ajan döngüsü (Open WebUI modeli: k2agent)
After=k2-swap.service
[Service]
Environment=PATH=$UPATH
ExecStart=$HOME/.local/bin/k2loop serve
Restart=on-failure
RestartSec=10
[Install]
WantedBy=default.target
UNIT
cat > "$UNITS/k2-phoenix.service" <<UNIT
[Unit]
Description=k2loop: Phoenix (ToT iz ağacı) http://127.0.0.1:6006
[Service]
Environment=PHOENIX_HOST=127.0.0.1
Environment=PHOENIX_PORT=6006
Environment=PHOENIX_GRPC_PORT=4317
Environment=PHOENIX_WORKING_DIR=$DATA/phoenix
# tamamen yerel: telemetri, dış kaynaklar ve Phoenix'in kendi ajan asistanı (sunucu bash'i, web, GitHub) kapalı
Environment=PHOENIX_TELEMETRY_ENABLED=false
Environment=PHOENIX_ALLOW_EXTERNAL_RESOURCES=false
Environment=PHOENIX_DISABLE_AGENT_ASSISTANT=true
Environment=PHOENIX_AGENTS_DISABLE_BASH=true
Environment=PHOENIX_AGENTS_DISABLE_WEB_ACCESS=true
Environment=PHOENIX_AGENTS_DISABLE_GITHUB=true
ExecStart=$HOME/.local/bin/phoenix serve
Restart=always
RestartSec=10
[Install]
WantedBy=default.target
UNIT
cat > "$UNITS/k2-ldr.service" <<UNIT
[Unit]
Description=k2loop: Local Deep Research http://127.0.0.1:5000
After=k2-swap.service
[Service]
Environment=LDR_WEB_HOST=127.0.0.1
Environment=LDR_WEB_PORT=5000
Environment=LDR_LLM_PROVIDER=openai_endpoint
Environment=LDR_LLM_OPENAI_ENDPOINT_URL=http://127.0.0.1:8080/v1
Environment=LDR_LLM_OPENAI_ENDPOINT_API_KEY=none
Environment=LDR_LLM_MODEL=helper
Environment=LDR_SEARCH_TOOL=searxng
Environment=LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL=http://127.0.0.1:8888
ExecStart=$HOME/.local/bin/ldr-web
Restart=on-failure
RestartSec=15
[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
# enable + restart: a re-run must apply new unit files / settings to services that are already running
start_unit() {
  systemctl --user enable "$1" >/dev/null 2>&1 && systemctl --user restart "$1" >/dev/null 2>&1 && ok "$1" \
    || warn "$1 başlatılamadı: journalctl --user -u $1 -n 50"
}
[ "$SKIP_LLM" -eq 0 ] && start_unit k2-swap.service
[ "$SKIP_LAYA" -eq 0 ] && start_unit k2-laya.service
command -v phoenix >/dev/null && start_unit k2-phoenix.service
[ "$WITH_WEB" -eq 1 ] && command -v ldr-web >/dev/null && start_unit k2-ldr.service
start_unit k2-agent.service
if [ "$SKIP_ROOT" -eq 0 ]; then
  sudo loginctl enable-linger "$USER" || warn "linger açılamadı (servisler sadece oturum açıkken çalışır)"
fi

# ---------------------------------------------------------------- 10. ayar + kontrol
say "10/10 Ayar dosyası ve kontrol"
[ -f "$CONF/config.yaml" ] || cp "$HERE/config/config.example.yaml" "$CONF/config.yaml"
if [ "$HYBRID" -eq 1 ]; then k2loop mode hybrid; else k2loop mode local; fi
[ "$WITH_LOCAL" -eq 1 ] && k2loop strong "$STRONG_KIND"
ok "$CONF/config.yaml"
if [ "$SKIP_LLM" -eq 0 ]; then
  echo "  Modeller yükleniyor; ilk seferde K2 için uygun context ölçülür (en fazla ~10 dk)..."
  for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8080/running 2>/dev/null | grep -q k2-horizon && break; sleep 5; done
fi
trap - ERR
k2loop doctor || true

cat <<MSG

KURULUM BİTTİ. Son adımlar:
  1) Yeni bir terminal açın (PATH güncellensin).
  2) Tarayıcı: http://127.0.0.1:3000  → Open WebUI'da ilk hesabı oluşturun → model olarak 'k2agent' seçin.
     Örnek mesaj:  "~/projem içine CSV'den grafik çizen bir komut ekle, testleriyle"
  3) (Önerilen) http://127.0.0.1:5000 → Local Deep Research hesabı oluşturun, sonra
     $SECRETS dosyasına LDR_USER=... ve LDR_PASSWORD=... satırlarını yazın:
        systemctl --user restart k2-agent
  4) İzleme:  ToT ağacı http://127.0.0.1:6006  ·  modeller http://127.0.0.1:8080/ui
  5) Tam test: k2loop doctor --full
  Loglar: journalctl --user -u k2-agent -f   |   journalctl --user -u k2-swap -f
MSG
