#!/usr/bin/env bash
# k2loop kaldırma. Projelerinize ve ~/.config/k2loop ayarlarınıza dokunmaz.
set -uo pipefail
DATA="${K2LOOP_DATA:-$HOME/.local/share/k2loop}"
UNITS="k2-agent k2-swap k2-laya k2-phoenix k2-ldr k2-llm k2-strong"
for u in $UNITS; do
  systemctl --user disable --now "$u.service" 2>/dev/null
  rm -f "$HOME/.config/systemd/user/$u.service"
done
systemctl --user daemon-reload
for c in k2-webui k2-searxng; do sudo docker rm -f "$c" >/dev/null 2>&1 && echo "docker: $c kaldırıldı"; done
echo "(Open WebUI sohbet geçmişi 'k2-webui' docker volume'unda duruyor: sudo docker volume rm k2-webui)"
rm -f "$HOME/.local/bin/llama-swap"
for t in k2loop bernstein aider-chat ruff arize-phoenix local-deep-research; do uv tool uninstall "$t" 2>/dev/null; done
read -r -p "Modeller, Laya logları, ajan geçmişi ve derlemeler ($DATA) silinsin mi? [e/H] " a
[ "${a,,}" = "e" ] && rm -rf "$DATA" && echo "silindi"
echo "Tamam. (CUDA toolkit ve docker kaldırılmadı.)"
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "WSL: Windows tarafında elle kaldırılacaklar:"
  echo "  - Görev Zamanlayıcı → 'k2loop WSL baslat' görevini sil"
  echo "  - %USERPROFILE%\\.wslconfig (k2loop için yazıldıysa) → sil, sonra: wsl --shutdown"
fi
