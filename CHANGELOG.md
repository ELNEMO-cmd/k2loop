# Değişiklikler

## Yayınlanmamış

### Yeni: kod hedefleri hedef bitene kadar döner
- Önceden bir adım başarısız olunca hedef bitiyordu. Artık **turlar** var: plan → adımlar → kontrol → kalanı yeniden
  planla. Yeni plan, commit'lenmiş adımları, başarısız denemeleri (ve sonuçlarını) ve eksikleri bilir.
- Bütün adımlar geçince güçlü model hedefi diff'e bakarak doğrular (`agent.verify_goal`); eksik bulursa yeni tur açılır.
- Öncelik işi bitirmek: tur ve süre sınırı varsayılan olarak yok (`agent.max_rounds`, `agent.goal_time_budget_min`:
  0 = sınırsız). Her başarısızlıktan sonra sonraki denemenin çabası artar (aday sayısı, hata ayıklama derinliği ve
  dal sayısı, güçlü-model eskalasyonu, görev süresi). Sadece `agent.max_stalled_rounds` (3) tur üst üste hiçbir adım
  geçmezse durup insana sorar. Hedef kontrolünün belirsiz cevabı tekrar sorulur, asla "bitti" sayılmaz.
  `agent.stop_on_failed_step` kaldırıldı (başarısız adım artık turu bitirir, hedefi değil).
- Open WebUI'da her tur `🔁 Tur n/N` başlığıyla, hedef kontrolü `🎯` / `🔁` satırıyla görünür.

### Düzeltmeler
- **Ajan döngüsü sessizce ölebiliyordu.** Planlayıcı hata durumunda `SystemExit` fırlatıyordu; ajan bunu yakalamadığı
  için döngü thread'i bitiyor, hedef sonsuza dek `running` kalıyor, yeni hedefler hiç çalışmıyordu. Artık hedef
  `failed` olur, döngü devam eder.
- **Model `.k2loop/check.sh`'i değiştirip testleri "geçebiliyordu".** Dosya artık aday diff'lerine girmez ve her
  kontrol, commit'li orijinal `check.sh` ile çalışır.
- **Model JSON dizi döndürünce çöküş.** Niyet sınıflandırma, review ve planlayıcı artık beklenmeyen JSON biçimlerini
  tolere eder (önceden testleri geçen bir aday bile review'da çöküp kayboluyordu).
- **Zaman aşımında yetim süreçler.** `check.sh` ve Aider kendi süreç grubunda çalışır; süre dolunca tüm grup öldürülür.
- **İptal sonrası commit.** `/iptal` (veya Bernstein SHUTDOWN) gelince yeni aday üretilmez ve kazanan commit'lenmez;
  patch kaydedilir.
- **Okunamayan review cevabı onay sayılıyordu.** Artık oy sayılmaz.
- Küçükler: `.gitignore` satır tekrarı, zaten `k2loop/` dalındayken anlamsız merge önerisi, boş mesajın hedef olması,
  `secrets.env` tırnakları, yerel modda yanlış kaynak etiketleri ("claude", "qwen27b"), Aider geçmiş dosyalarının
  sınırsız büyümesi, `install.sh`'in geçersiz `--strong` değerini en sonda fark etmesi ve `k2-agent`'ı ayarlar
  yazılmadan başlatması, eskimiş "Qwen3.8 27B varsayılan" metinleri.

## 0.2.0 — 2026-09-26

Tamamen yerel, sürekli çalışan ajan. Gerçek donanımda (RTX 3080 Ti, WSL2 Ubuntu 24.04) uçtan uca test edildi:
bir kod hedefi (2 adım, ToT hata ayıklaması dahil) ve bir plan hedefi (tartışma + hakem) başarıyla tamamlandı.

### 0.1'in "pek çalışmamasının" sebepleri — düzeltildi
- **K2 Horizon, Aider'ın düzenleme biçimini kullanmıyordu.** Kendi araç-çağrı sözdizimiyle (`<ifm|tool_call>…`) ya da
  tam dosya bloklarıyla cevap veriyordu; Aider hiçbirini uygulayamadığı için her aday boş kalıyordu. Yeni
  `k2adapter` bu cevapları Aider'ın SEARCH/REPLACE biçimine çevirir.
- **Araçlar PATH'te yoktu.** systemd / Bernstein ile başlayan süreçler `~/.local/bin`'i görmüyordu → "aider not found",
  "ruff: command not found". k2loop artık araç dizinlerini kendisi ekler.
- **Güçlü model PCIe üzerinden sürünüyordu.** WSL'de Windows NVIDIA sürücüsü VRAM taşınca hata vermek yerine sistem
  RAM'ine taşıyor; "kaç katman sığar" testi yanılıp 16 GB modeli 12 GB karta "sığdırıyordu". GPU yerleşimi artık
  llama.cpp'nin `--fit` ayarında.
- K2'nin `</ifm|think_faster>` düşünme etiketleri cevaplardan temizlenmiyordu.
- Küçük model hiçbir dosyayı değiştirmediğinde eskalasyon mesajı "testler kalıyor" diyordu (yanıltıcı).

### Yeni
- **`k2loop serve`**: sürekli hedef kuyruğu + Open WebUI için OpenAI uyumlu uç (`k2agent`); her adım canlı akar.
- **Üç hedef türü:** araştırma · plan (ToT dalları + çok-ajanlı tartışma + hakem + son düzenleme) · kod
  (önce internette hazır çözüm ara → ToT plan: hazırı kullan / uyarla / sıfırdan → ToT + Self-Consistency kodlama).
- **llama-swap**: tüm modeller tek adreste; GPU'da K2 ⇄ güçlü model, CPU'da yardımcı model (Qwen3 4B) hep açık.
- **Güçlü model MoE oldu:** Qwen3.6-35B-A3B (~21 token/sn; eski Qwen3.8 27B ~1.5–4 token/sn). `k2loop strong moe|dense`.
- **SemIf yöntemi**: yardımcı modelden tek geçişte seçenek olasılıkları; Laya'nın yanında ikinci gölge tahminci.
  `k2loop laya-eval` kural / Laya / SemIf'i karşılaştırır.
- GPU/CPU üst üste bindirme: K2 bir sonraki adayı yazarken öncekinin testleri CPU'da koşar.
- Phoenix ile ToT iz ağacı (telemetri ve Phoenix'in kendi ajan asistanı kapalı), Local Deep Research + SearXNG.
- WSL2 desteği: iki parçalı kurulum (`--root-part` / `--skip-root`), WSL'i ayakta tutan oturum görevi.
- Tamamen yerel varsayılan (Claude yalnızca `--hybrid` ile).

### Bilinen eksikler
- Araştırma, Local Deep Research hesabı bağlanmadan yalnızca SearXNG + küçük model özetiyle çalışır.
- Laya ve SemIf gölge modda; `laya-eval` için ~150 etiketli karar gerekiyor.
- K2'nin Türkçesi zayıf; plan hedefleri bu yüzden varsayılan olarak güçlü modelde yazılır.
