# Gas Sensor Array Drift — BN ve LN Karşılaştırması

Bu proje, UCI Gas Sensor Array Drift veri seti üzerinde **Standart YSA**, **Batch Normalization (BN)** ve **Layer Normalization (LN)** mimarilerini karşılaştırır.

## Deney düzeni

- 10 adet `batch1.dat`–`batch10.dat` dosyası, toplam 13.910 örnek ve 128 özellik
- Sınıf oranları korunarak rastgele %70 eğitim / %15 doğrulama / %15 test
- `StandardScaler` yalnızca eğitim verisine uyarlanır
- 128→128→64→6 mimarisi; ReLU; BN/LN iki gizli katmanın ardından uygulanır
- Adam, öğrenme oranı 0.001, batch size 64, 100 epoch
- 3 tekrar: 42, 123, 2026; CPU üzerinde eğitim
- En düşük doğrulama kaybına sahip ağırlıklarla test

**Not:** Veri ayırımı örnek bazında rastgeledir; batch'ler arasında kronolojik/genelleme testi değildir. Rapor sonuçları bu protokole aittir.

## Kurulum

Python 3.10+ önerilir.

```bash
python -m pip install -r requirements.txt
```

Veri setini [UCI Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) sayfasından temin edin. On adet `.dat` dosyasını `data/` klasörüne yerleştirin.

```bash
python main.py
```

Farklı veri klasörü kullanmak için:

```bash
python main.py --data-dir "C:/veri_klasoru"
```

Çalıştırma sonunda `figures/`, `tables/` ve `results/` dizinleri üretilir. Eğitim dokuz model/tekrar kombinasyonu içerir ve bilgisayara göre zaman alabilir.

## Dosyalar

- `main.py`: komut geçmişinden derlenmiş veri analizi, eğitim, test ve grafik kodu
- `requirements.txt`: gerekli Python paketleri
- `data/`: kullanıcı tarafından eklenecek veri seti
- `results/`: kullanıcı tarafından eklenecek **orijinal deney çıktıları** veya yeniden çalıştırma çıktıları
- `figures/`: grafikler
- `tables/`: CSV tablolar

## Tekrarlanabilirlik ve sınırlar

Bu sürüm, kullanıcının komut geçmişindeki model tanımlarını ve deney ayarlarını korur. Kod sözdizimi kontrol edilmiştir ancak bu pakette veri dosyaları olmadığı için uçtan uca eğitim çalıştırılmamıştır. Yeniden eğitim yapılırsa çalışma ortamına bağlı olarak sayısal sonuçlar farklılık gösterebilir. Raporla birebir aynı sonuçların paylaşılması için özgün `results/` ve `figures/` çıktıları ayrıca eklenmelidir.
