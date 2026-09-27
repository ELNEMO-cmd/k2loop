# k2loop test raporu — 2026-09-27

Test: `claude/sorun-analizi-q3nfvi` dalı (bench + hedef döngüsü) gerçek bir görevle, tamamen yerel çalıştırıldı.

- Donanım: RTX 3080 Ti 12 GB, WSL2. Modeller: K2 Horizon 7B (kod), Qwen3.6-35B-A3B (plan / eskalasyon / review).
- Görev (`k2loop ask` ile): `~/borsa-agent` içinde, haberleri ve fiyatları izleyip hisse alıp satan bir borsa ajanı.
  Bileşenler: yfinance fiyat verisi, RSS haberleri + TR/EN duygu skoru, SMA20/50 + RSI14, risk kuralları, SQLite'ta
  kağıt üzerinde işlem (paper trading), CLI (`run --once` / `report` / `backtest`), internetsiz pytest testleri.
- Düzeltmeler (bu rapordaki "düzeltildi" maddeleri) `k2-adapter-fixes` dalında. Test sayısı: 80 → 100, hepsi geçiyor.
- Açık kalan sorunlar GitHub Issue olarak açıldı (bu dosyanın sonunda liste).

## Sonuç (kısa)

Kısmen başarılı. Aday → kontrol → review mimarisi işe yarıyor: review gerçek hatalar yakaladı (ters nakit kontrolü,
hiç tetiklenmeyen stop-loss). Ama sistem gerçek bir görevde ilk kez sınandı ve yapıştırıcı kodda (Aider çağrısı,
`check.sh`, K2 adapter'ı, aday seçimi) çok sayıda hata çıktı. Düzeltmelerden önce, yeni dosya gerektiren adımlarda
K2'nin üç adayı da boş dönüyordu. Düzeltmelerden sonra kolay/orta adımlar çoğunlukla ilk iki denemede geçiyor; zor
adımlar ancak 35B'ye devirle çözülüyor.

Süre: yaklaşık 12 saatte 7 adım commit'lendi (config, fiyat, göstergeler, haber/duygu, Google News, broker).
Bunun yaklaşık 6 saati, düzeltmeleri yüklemek için ajanın 13 kez yeniden başlatılmasından ve her seferinde ~30 dk
yeniden planlamadan geldi (bkz. açık sorun: yeniden başlatınca kaldığı yerden devam edemiyor).

Borsa ajanının durumu: veri tarafı testli ve commit'li. Strateji, ajan döngüsü, CLI ve backtest devam ediyor.
Gerçek aracı kuruma emir gönderen kod yok. 7B modelin yazdığı alım-satım kodu insan incelemesinden geçmeden gerçek
parayla kullanılmamalı.

## Düzeltilen hatalar (k2-adapter-fixes dalı)

### Aider / K2 adapter'ı
1. **Aider, sohbette olmayan dosya adı geçen cevabın düzenlemelerini atıyor.** Aider (`base_coder.py`,
   `check_for_file_mentions`) dosyayı ekleyip tekrar soruyor ve ilk cevabı uygulamıyor; K2 tekrar edemiyor.
   Düzeltme: proxy, "I added these files to the chat" isteğine önceki cevabı aynen döndürüyor (`k2adapter.replay_edits`).
2. **K2'nin düz JSON düzenlemeleri** (`{"files":[{"path","search","replace"}]}`) tanınmıyordu. Artık SEARCH/REPLACE'e çevriliyor.
3. **Kural metnindeki `bash .k2loop/check.sh`**, Aider'ın check.sh'i her çalıştırmada sohbete eklemesine yol açıyordu.
   Aider'a giden mesajlarda artık dosya yolu yok (`engine.CHECKS`).
4. **Boş kod bloğu bir sonraki dosyanın adını yutuyordu.** Örneğin `borsa_agent/__init__.py` içine `tests/__init__.py`
   satırı yazıldı. Kod bloğu regex'i düzeltildi.
5. **K2'nin bozuk blokları bütün cevabı reddettiriyordu.** `>>>>>>> REPLACE` eksik ya da yarım `<<<<<<< SEARCH`
   bırakılınca Aider tüm cevabı reddediyordu. Eksik işaret artık tamamlanıyor, yarım bloklar atılıyor.
6. **K2 kodu düşünme kısmına yazıp cevaba boş iskelet bırakıyordu.** Artık yalnızca YENİ dosyalar
   `reasoning_content`'ten alınıyor (`path:` + boş satır ya da `# path` biçimleri); var olan dosyalara dokunulmuyor.
7. **Aider, görev metnindeki URL'yi kendiliğinden indiriyordu.** Playwright'ı `sudo` ile kurmaya çalıştı, bağlam 37k
   token'a şişti ve 25 dk zaman aşımına kadar asılı kaldı. Artık `--no-detect-urls` veriliyor.
8. **Aider'ın 600 sn API zaman aşımı, 35B eskalasyonlarını tam 10:00'da kesiyordu** (llama-swap: 499). Artık `--timeout` veriliyor.

### check.sh / aday seçimi
9. **Projede test yokken adaylar hiç sınanmadan "test GEÇTİ" sayılıyordu** ve en küçük diff (testsiz aday) kazanıyordu.
   Artık adayın kendi eklediği testler çalıştırılıyor; testleri gerçekten çalışan aday öne geçiyor.
10. **`requirements.txt` bağımlılıkları kurulmuyordu** (`uv run --with pytest`); doğru adaylar `ModuleNotFoundError: yaml`
    ile eleniyordu. Artık `--with-requirements` kullanılıyor.
11. **ruff 0.16'nın varsayılan stil kuralları** (UP006, RUF012…) adayları eliyordu. Projede ruff ayarı yoksa artık
    `--select E4,E7,E9,F` kullanılıyor.
12. **pytest `-x` ile çalışıyordu:** hata sayısı hep 1 çıkıyor, engine ilerlemeyi göremiyordu. `-x` kaldırıldı.
13. **Lint hatası testleri gizliyordu:** lint'te kalan aday "1 hata" sayılıp 5 test hatalı adayı yeniyordu. 35B'ye de
    "sadece lint hatası" diye devredildi; model gerçek düzeltmenin ihtiyaç duyduğu değişkeni sildi. Lint ve tip kontrolü
    artık testleri durdurmuyor.
14. **Sadece boş dosya oluşturan aday "kullanılabilir" sayılıyordu.** Boş `agent.py` testleri geçip "Implement Agent"
    adımında kazandı. Artık en az bir değişen satır gerekiyor.
15. **Hata ayıklama turlarında yeni dosyalar Aider'a verilmiyordu:** dosya listesi adım başında hesaplanıyordu ve mesaj
    hâlâ "yeni dosya" diyordu. Artık her worktree için ayrı hesaplanıyor.
16. **Yeni dosya gerektiren adımlarda K2 kod yazmak yerine "dosyaları ekler misiniz" diye soruyordu.** Görevde yeni
    dosyalar artık açıkça belirtiliyor.
17. **Aider önbelleği (`.aider.tags.cache.v4/*.db-wal`, ~400 KB ikili) commit'lere giriyordu.** Hariç tutma deseni düzeltildi.

### Planlama / döngü / diğer
18. **Hedef döngüsü varsayılan ayarlarla durmayabiliyordu:** doğrulayıcı her tur yeni bir eksik bulursa sonsuz döngü
    (test: 30 tur, 30 commit). Eklenen: `agent.max_verify_rounds` (6).
19. **Planlayıcı tamamlanmış işi baştan planlıyordu** (hakem "From scratch" planını seçti). Plan istemine "yalnızca
    eksik işi planla" kuralı, hakeme de depodaki dosya listesi eklendi.
20. **SearXNG motorları engelliyken araştırma sessizce "0 sonuç" diyordu.** Artık anahtar kelimelerle yeniden deneniyor
    ve engelleyen motorlar mesajda adıyla belirtiliyor.
21. **Lint hataları "assertion_logic" (mantık hatası) diye sınıflanıyordu;** hata ayıklama ipuçları yanlış yöne gidiyordu.
22. **`No module named 'bourse_agent'`** (depoda `borsa_agent` var): modeller bunu yol sorunu sanıp `conftest.py`
    ekliyordu. Artık "muhtemelen yazım hatası" ipucu veriliyor.
23. Arayüz: değişiklik yapmayan aday ✅ gösteriliyordu; süreler duvar saatiyle ölçüldüğü için negatif çıkıyordu (−0,9 sn).

## Açık sorunlar (düzeltilmedi — GitHub Issue)

Her biri ayrı bir Issue olarak açıldı: belirti, kanıt, kök neden, önerilen çözüm.

- [#2](https://github.com/ELNEMO-cmd/k2loop/issues/2) Review'da oybirliğiyle reddedilen aday, düzeltme başarısız olunca yine de commit'leniyor
- [#3](https://github.com/ELNEMO-cmd/k2loop/issues/3) Adım kabulünde "yeni davranış için test var mı?" kontrolü yok
- [#4](https://github.com/ELNEMO-cmd/k2loop/issues/4) K2 düşünmesini cevaba döküyor (taslak bloklar + son JSON); adapter güvenle ayıklayamıyor
- [#5](https://github.com/ELNEMO-cmd/k2loop/issues/5) K2 zaman zaman token sınırına kadar döngüye giriyor (~10 dk, kullanılamaz cevap)
- [#6](https://github.com/ELNEMO-cmd/k2loop/issues/6) Ajan yeniden başlayınca hedefe kaldığı yerden devam edemiyor (her seferinde ~30 dk yeniden planlama)
- [#7](https://github.com/ELNEMO-cmd/k2loop/issues/7) install.sh'in SearXNG ayarında yalnızca 3 web motoru var; kısa sürede engelleniyorlar
- [#8](https://github.com/ELNEMO-cmd/k2loop/issues/8) Eski şablonla üretilmiş check.sh'ler "kullanıcı düzenlemiş" sayılıyor ve düzeltmeleri hiç almıyor
- [#9](https://github.com/ELNEMO-cmd/k2loop/issues/9) Aider'ı başarısız olan (`aider_ok=False`) adayın diff'i yine aday olarak değerlendiriliyor
- [#10](https://github.com/ELNEMO-cmd/k2loop/issues/10) Bench adil değil: replay/onarım yalnızca K2 yolunda, 35B doğrudan Aider'a gidiyor
- [#11](https://github.com/ELNEMO-cmd/k2loop/issues/11) Config yolu sabit (`~/.config/k2loop`): ikinci bir ajanı (test / yeni sürüm) yan yana çalıştırmanın yolu yok

## Son durum (test durduruldu)

Test, görev bitmeden kullanıcı isteğiyle durduruldu.

- Borsa ajanı: https://github.com/ELNEMO-cmd/borsa-agent (gizli), dal `k2loop/20260927-0303-borsa-agent-icinde-python-ile-`.
  Commit'li ve testli (16 test geçiyor): config, fiyat verisi (yfinance), SMA/RSI, haber + TR/EN duygu skoru
  (Google News RSS dahil), SQLite'ta kağıt üzerinde broker. Broker testsiz kabul edildi (#2, #3).
- Yarım kalan: strateji modülü. Hata ayıklama + review düzeltmesiyle testleri geçen bir aday vardı (review'ın bulduğu
  ters nakit kontrolü ve hiç tetiklenmeyen stop-loss düzeltilmişti), ama ajan commit'ten önce durduruldu ve o
  worktree kayboldu. Aynı hedef tekrar verilirse bu adımdan başlar (bkz. #6).
- Yapılmadı: alım-satım ajanı döngüsü, CLI (`run --once`, `report`, `backtest`), backtest, README.
- Kodda bilinen kusur: 1. adımda K2'nin yazdığı docstring'lerde `bourse_agent` yazım hatası var; sonraki adımlarda
  modeller bunu import'lara kopyaladı.

## 2. çalıştırma (22:20'de yeniden başlatıldı)

Aynı hedef, aynı dal (`k2-adapter-fixes`), yeniden başlatmadan sonra tekrar verildi.

- Planlama bu kez yalnızca kalan işi kapsadı (4 adım: strateji, ajan döngüsü, CLI, README). 19 numaralı düzeltme işe yaradı.
- Borsa ajanındaki `bourse_agent` yazım hatası kullanıcı isteğiyle elle düzeltildi (borsa-agent `6772993`).
- Strateji adımı: K2'nin 8 denemesinin hiçbiri geçmedi, adım 35B'ye devredildi. Denemelerden 2'si boş kaldı,
  2 düzeltme turu da hiçbir şeyi değiştirmedi.
- Yeni açık sorunlar:
  - [#12](https://github.com/ELNEMO-cmd/k2loop/issues/12) K2 adapter: K2'nin kodu kayboluyor (yarım blok + reasoning, `finish_reason=length`, boş cevap)
  - [#13](https://github.com/ELNEMO-cmd/k2loop/issues/13) Hiçbir şey değiştirmeyen düzeltme denemesi yine aday sayılıyor ve deneme hakkı yiyor
