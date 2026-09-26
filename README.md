# k2loop 0.2 — tamamen yerel, sürekli çalışan ToT ajanı

Tek RTX 3080 Ti (12 GB) + 32 GB RAM için. Hedefi yazarsın; ajan **önce internette hazır çözüm arar**, sonra
**Tree of Thoughts / Self-Consistency** ile planlayıp kodlar. Ne yaptığını Open WebUI'da canlı izlersin.
Hiçbir şey bilgisayardan çıkmaz (internete sadece arama için çıkılır).

```
 Open WebUI :3000 (canlı ajan)   Phoenix :6006 (ToT iz ağacı)   LDR :5000 (araştırmalar)   llama-swap :8080/ui
        │ OpenAI API                     ▲ OpenTelemetry              ▲                          │
   k2loop serve :8090 — hedef kuyruğu: ARAŞTIR → PLANLA (ToT) → KODLA (ToT+SC) → REVIEW            │
        │                                   │                          │                         │
   GPU (3080 Ti) ─ llama-swap ─────────────────────────────────────────────────────────────────────┘
     K2 Horizon 7B: kod adayları (Aider)  ⇄  güçlü model: plan / eskalasyon / son review
   CPU ─ yardımcı model (Qwen3 4B, hep açık) · Laya + SemIf (kararlar) · check.sh · MoE uzmanları · SearXNG · LDR
```

| Parça | Görevi | Nerede |
|---|---|---|
| **K2 Horizon 7B** | Kod adayları, fix hipotezleri, review oyları | GPU |
| **Güçlü model** | Plan (3 alternatif + hakem), eskalasyon, son review. `k2loop strong moe\|dense` | GPU+RAM, gerektiğinde K2'nin yerine |
| ↳ `moe`: **Qwen3.6-35B-A3B** UD-IQ4_XS | 35B parametre, token başına 3B aktif: uzmanlar RAM'de/CPU'da, gerisi GPU'da → hızlı | önerilen |
| ↳ `dense`: **Qwen3.8 27B** Q4_K_M | 12 GB'a sığmaz, katmanların bir kısmı CPU'da → ~4 token/sn | eski varsayılan |
| **Yardımcı model** (Qwen3 4B) | Hedefin türü, hipotez puanlama, araştırma özetleri, Open WebUI başlıkları | CPU, K2 ile aynı anda |
| **Laya** | Test çıktısı sınıflandırma, ToT dal budama, tekrar-dene/eskale-et | CPU |
| **llama-swap** | Tüm modeller tek adreste; GPU'da K2⇄Qwen takası, CPU modeli hep açık | yerel |
| **Local Deep Research + SearXNG** | Çok turlu internet araştırması, kaynaklı rapor, zamanlanmış abonelikler | CPU |
| **Open WebUI** | Sohbet arayüzü; ajanın her adımı canlı akar | docker |
| **Phoenix** | Her aday / dal / Laya kararı / model çağrısı iç içe ağaç olarak | yerel |
| **Aider** | K2'nin dosya düzenleme aracı | yerel |
| **Bernstein 3.20.0** | (isteğe bağlı) `k2loop go` ile çok görevli planları paralel worktree'lerde çalıştırır | yerel |

## Kurulum

```bash
git clone https://github.com/<kullanıcı>/k2loop.git && cd k2loop
./install.sh                 # güçlü model varsayılan olarak MoE (--strong dense: eski Qwen3.8 27B)
```

Güncellemek için: `git pull && ./install.sh --skip-root` (yapılmış adımlar atlanır, servisler yeni ayarlarla
yeniden başlar). Değişiklik geçmişi: [CHANGELOG.md](CHANGELOG.md).

0.1'in üstüne kurulabilir: eski `k2-llm`/`k2-strong` servisleri kapatılır, modeller llama-swap'a taşınır, indirilmiş
modeller ve Laya logları korunur. Süre 30–60 dk, ~55 GB disk. Log: `~/k2loop-install.log`.

Kurulumdan sonra:
1. `http://127.0.0.1:3000` → Open WebUI'da ilk hesabı oluştur → model: **k2agent**.
2. (Önerilen) `http://127.0.0.1:5000` → Local Deep Research hesabı oluştur; bilgileri
   `~/.config/k2loop/secrets.env` dosyasına `LDR_USER=...` ve `LDR_PASSWORD=...` olarak yaz,
   `systemctl --user restart k2-agent`. (Yazmazsan araştırma basit SearXNG aramasına düşer.)
3. `k2loop doctor --full` → her şey ✓ olmalı.

Seçenekler: `--strong moe|dense` (güçlü model; önerilen `moe`) · `--no-local` (güçlü modelsiz) · `--no-web`
(SearXNG/LDR/Open WebUI'sız) · `--hybrid` (Claude Code'lu eski mod) · `--quant Q5_K_M`.

### Windows'ta (WSL2) kurulum

Aynı paket WSL2'deki Ubuntu 24.04'te çalışır (NVIDIA GPU WSL'den kullanılabilir):
1. Yönetici terminalde `wsl --install -d Ubuntu-24.04`, yeniden başlat, Ubuntu kullanıcı adı/şifresini belirle.
2. `%USERPROFILE%\.wslconfig`: `[wsl2] memory=24GB swap=8GB` (varsayılan RAM'in yarısıdır; güçlü model için az).
3. sudo şifresi sormadan iki parça kurulum (Windows'tan):
   `wsl -u root -- ./install.sh --root-part --for-user KULLANICI` → `wsl -- ./install.sh --skip-root --strong moe`.
4. WSL terminal kapanınca kendini kapatır: Görev Zamanlayıcı'da oturum açılışında
   `wsl.exe -d Ubuntu-24.04 --exec /bin/sleep infinity` çalışan bir görev WSL'i (ve ajanı) ayakta tutar.
5. Arayüzler Windows tarayıcısından aynı adreslerle açılır (`http://127.0.0.1:3000` …).

**Önerilen Windows ayarı:** NVIDIA Denetim Masası → 3D Ayarları → *CUDA - Sysmem Fallback Policy* →
**Prefer No Sysmem Fallback**. Varsayılanda sürücü VRAM taşınca hata vermek yerine sessizce sistem RAM'ine taşır ve
model PCIe üzerinden çok yavaş koşar. (k2loop artık GPU katmanlarını llama.cpp'nin `--fit` ayarına bıraktığı için
buna düşmez, ama başka programlar için de iyi bir ayardır.) Ajan 7/24 çalışacaksa Windows'ta prizdeyken uykuyu kapat.

## Kullanım

Open WebUI'da **k2agent** modeline yaz:

- `~/projem içine CSV'den grafik çizen bir komut ekle, testleriyle` → **kod** hedefi
- `yerel LLM'ler için en iyi kod arama araçlarını karşılaştır` → **araştırma** hedefi
- `3 ayda Almanca B1 için çalışma planı yap` → **plan** hedefi (her konu olur)
- `/durum` · `/iptal` · `/izle` (çalışan hedefe bağlan) · `/yardim`

Bir **plan** hedefi şöyle ilerler: konu önce araştırılır → 3 planlayıcı (Pragmatik, Kapsamlı, Risk-öncelikli) farklı
açılardan plan yazar (K2) → **birbirleriyle tartışırlar**: her biri diğerlerinin planını okur, sohbette görünen kısa bir
mesajla eleştirir, öncekilerin mesajlarına cevap verir ve kendi planını düzeltir → hakem (Qwen) puanlar → kazanan plan
diğerlerinin iyi fikirleriyle son haline getirilir. Tartışma dökümü ve tüm adaylar `~/.local/share/k2loop/agent/plans/`
altında. Ayarlar: `config.yaml` → `plan.*` (planlayıcı sayısı, tartışma turu, hangi model).

Tarayıcıyı kapatsan da hedef çalışmaya devam eder; birden fazla mesaj yazarsan sıraya girer. Terminalden:
`k2loop ask "hedef"` (aynı canlı akış). Proje yolu yazmazsan `config.yaml` → `agent.project` kullanılır.

Bir kod hedefi şöyle ilerler:
1. **Araştırma:** "Bunu yapan hazır proje/kütüphane var mı, lisansı ne, ne alınabilir?" → rapor `.k2loop/plans/*-research.md`.
2. **Plan (ToT):** Qwen üç yol üretir: *hazırı kullan* / *uyarla* / *sıfırdan yaz*; hakem seçer.
3. **Her adım (ToT + Self-Consistency):** K2 3 aday yazar (biri yazılırken öncekinin testleri CPU'da koşar) → geçenlerden
   en küçük diff; hiçbiri geçmezse hata sınıflandırılır (Laya #1) → 3 fix hipotezi → yardımcı model puanlar → 2'si seçilir
   (Laya #2) → denenir; seviyeler arasında devam/eskale/insana sor (Laya #3); gerekirse Qwen eskalasyonu.
4. **Review:** 3 oy + Qwen son kontrol. Kazanan diff `k2loop/<tarih>-<hedef>` dalına commit'lenir. Ana dalına sen alırsın.

Ajan projede commit'lenmemiş değişiklik varsa başlamaz; `.k2loop/check.sh` yoksa üretir. **`.k2loop/check.sh`
sistemin hakemidir** — istediğin gibi düzenle ve commit'le.

Eski komutlar da duruyor: `k2loop init`, `k2loop solve "tek görev"`, `k2loop plan/run/go` (Bernstein), `k2loop mode`,
`k2loop laya-eval`, `laya-export`, `laya-mode`.

## İzleme

| Ne | Nerede |
|---|---|
| Ajanın canlı akışı | Open WebUI `http://127.0.0.1:3000` |
| ToT ağacı, model çağrıları, süreler | Phoenix `http://127.0.0.1:6006` (proje: k2loop) |
| Hangi model yüklü, token hızları, sunucu logları | llama-swap `http://127.0.0.1:8080/ui` |
| Araştırma raporları, abonelikler (sürekli araştırma) | Local Deep Research `http://127.0.0.1:5000` |
| Ham olay kaydı | `~/.local/share/k2loop/agent/events.jsonl` |
| Bir görevin tüm dosyaları | `~/.local/share/k2loop/runs/<zaman>-.../` (aider logları, check çıktıları, patch'ler) |

## Kararlar: kural vs Laya vs SemIf

ToT hata ayıklamasındaki 3 karar (hata türü, dal budama, devam/eskale/insan) için üç aday var:
- **kural**: deterministik, başlangıçta kararı o verir;
- **Laya**: eğitilmiş karar modeli (CPU servisi);
- **SemIf** yöntemi ([SemIf](https://github.com/theoleecj/semif)): CPU'daki yardımcı model seçenekleri harflerle görür,
  tek geçişte her seçeneğin olasılığı okunur (metin üretimi yok).

Başlangıçta hepsi **gölge modda**: kural karar verir, Laya ve SemIf aynı kararı tahmin edip loglar; sonuçlar (dal testi
geçti mi, eskalasyon işe yaradı mı) otomatik etiket olur. `k2loop laya-eval` üçünü karşılaştırır (doğruluk + eşik
taraması); `--apply` her karar için kuralı en çok geçeni devreye alır (en az 150 etiket). Elle:
`k2loop laya-mode retry_escalate semif` · `... active` (Laya) · `... shadow` (kural). `k2loop laya-export` fine-tune verisi.

## Ayarlar

`~/.config/k2loop/config.yaml` (her hedef yeniden okur). Önemliler: `tot.*` (aday/dal/oy sayısı, süre, `overlap_checks`),
`roles.*` (yan işler GPU'da mı CPU'da mı), `agent.*`, `research.*`. Modellerin komut satırları:
`~/.config/k2loop/llama-swap.yaml` (değiştirince `systemctl --user restart k2-swap`). Anahtarlar: `secrets.env`.

## Sorun giderme

- **Modeller:** `journalctl --user -u k2-swap -f` veya `http://127.0.0.1:8080/ui`. K2 context'ini yeniden ölçmek için:
  `rm ~/.local/share/k2loop/llm_state.env && systemctl --user restart k2-swap`. Monitör 3080 Ti'ye takılıysa masaüstü de
  VRAM yer; mümkünse anakarta (iGPU) tak.
- **Ajan:** `journalctl --user -u k2-agent -f`; çökme ayrıntısı `~/.local/share/k2loop/agent/crash-*.txt`.
- **Open WebUI k2agent'ı görmüyor:** Ayarlar → Bağlantılar → `http://127.0.0.1:8090/v1`, anahtar `secrets.env`'deki
  `AGENT_API_KEY`.
- **Araştırma boş dönüyor:** `docker ps` (k2-searxng çalışıyor mu?), `systemctl --user status k2-ldr`, secrets.env'deki LDR hesabı.
- **CPU çok yükleniyor:** yardımcı modelin iş parçacığı: `llama-swap.yaml`'da `K2LOOP_HELPER_THREADS=6` ekle.

## Bilinen sınırlar

- 7B model küçük ve odaklı adımlarda iyidir; planlayıcı adımları bilerek küçük tutar.
- K2 Horizon, Aider'ın düzenleme isteklerine çoğu zaman kendi araç-çağrı biçimiyle (`<ifm|tool_call>…`) ya da tam
  dosya bloklarıyla cevap verir; Aider bunları uygulayamaz. `k2adapter` K2 ile Aider arasına girip cevapları Aider'ın
  SEARCH/REPLACE biçimine çevirir (`config.yaml` → `aider.k2_adapter: true`). Kapatılırsa K2 adayları boş kalır.
- Qwen her yüklendiğinde K2 GPU'dan iner (birkaç dakika). Qwen yarı RAM'de olduğu için yavaştır; görev bütçesi 200 dk.
  İkinci ekran kartında `llama-swap.yaml`'da ikisini ayrı gruplara koyup takası kaldırabilirsin.
- İnternetten gelen metin güvenilmez veri olarak işaretlenir ve planlayıcıya öyle verilir; yine de ajanın ürettiği dalları
  birleştirmeden önce incele.
- Embedded projelerde doğrulama derleme + host testleriyle sınırlıdır.
- K2 için llama.cpp'nin MBZUAI-IFM fork'u kullanılır. Bernstein beta yazılımdır (3.20.0'a sabit).

Kaldırma: `./uninstall.sh`
