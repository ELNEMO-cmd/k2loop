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
- **Takılınca araştırır:** başarısız adımın hata mesajı ve tekrar eden eksikler internette aranır, sonuç yeni plana
  girer (`agent.research_on_failure`).
- **En yakın başarısız deneme korunur:** diff'i ve hata çıktısı sonraki turun planına verilir.
- **Kodu yazan model ölçülerek seçilir, büyüklüğe göre değil.** Her aday `model_stats.jsonl`'a kaydedilir (model adı,
  proje türü, geçti mi, süre). `tot.codegen_profile: auto`: ölçülen geçme oranı daha iyi olan yazar (önce aynı proje
  türü). Takılınca: açıkça daha iyi olan (`tot.coder_margin`), veri yoksa/fark küçükse takılandan farklı olan.
  `tot.coder_min_samples` (20) aday birikene kadar varsayılan K2. Yeni komut: `k2loop model-stats`. Open WebUI'da her
  görevde "✍️ Kodu yazan: … (neden)" satırı.
- **Plan ⇄ kontrol sonsuz döngüsü kapatıldı:** sadece daha önce söylenmiş eksiklerin tekrar gelmesi ilerleme sayılmaz;
  `agent.max_stalled_rounds` tur üst üste ilerleme olmazsa insana sorulur.
- **Test yoksa:** uyarı + ilk adım test altyapısı; k2loop'un ürettiği `check.sh` yeni testleri görünce yenilenir.
- Open WebUI'da her tur `🔁 Tur n/N` başlığıyla, hedef kontrolü `🎯` / `🔁` satırıyla görünür.

### Yeni: `k2loop bench` — K2 vs 35B, gizli testlerle
- 8 gerçekçi Python görevi (kolaydan zora), her birinde modelin görmediği gizli testler ve referans çözüm. CI, her
  görevin anlamlı olduğunu doğrular (başlangıç kodu gizli testlerden kalır, referans çözüm geçer).
- Deneme = temiz repo + tek Aider çalıştırması (K2 gerçek k2adapter'dan geçer) + gizli testler; ToT motoru bilerek
  kullanılmaz. Rapor: görev bazında geçme/süre, %95 Wilson aralığı, "fark anlamlı mı". Sonuçlar `model_stats`'a yazılır.
- Gerçek Aider 0.86 ve sahte bir model sunucusuyla uçtan uca denendi (K2 araç-çağrı biçimi dahil).

### Düzeltmeler
- **Sonraki turların planı modelin context'ini taşırabiliyordu.** Tur hedefi (geçmiş + başarısız diff + araştırma)
  bütçeye sayılmıyordu; ölçümde 32–36 bin karakterlik istem 20,5 bin karakterlik bütçeye gidiyordu → planlama
  çağrısı başarısız → döngü duruyordu. Hedef artık bütçeye sayılır ve gerekirse baştan/sondan korunarak kısaltılır;
  hakem ve hedef kontrolü istemleri de bütçeye göre kesilir.
- **Hibrit modda hedef kontrolü yerel modeli arıyordu** (kurulu olmayabilir → her hedef "doğrulanamadı"). Artık
  hibritte Claude kontrol eder.
- **Model istatistikleri 35B aleyhine taraflıydı:** eskalasyonlar (sadece güçlü model alır, K2'nin çözemediği işler)
  ve hata-düzeltme denemeleri de sayılıyordu. Seçimde artık sadece taze adaylar ve bench sayılır.
- **GPU ping-pong:** kodu güçlü model yazarken hata hipotezleri ve review hâlâ K2'ye gidiyordu → her adımda birkaç
  kez model takası. Artık kodu yazan model bunları da yapar (CPU'daki yardımcı model aynen kalır).
- Tekrar eden eksik tespiti "add tests" gibi kısa bir maddeyi daha uzun, farklı bir maddeyle aynı sayabiliyordu.
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
