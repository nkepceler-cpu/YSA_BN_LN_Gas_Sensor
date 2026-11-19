"""BN/LN/Standart YSA karşılaştırması — UCI Gas Sensor Array Drift.
Kaynak: kullanıcının Python komut geçmişindeki deney adımları.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

parser = argparse.ArgumentParser(description="YSA BN/LN deneylerini çalıştır")
parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
args = parser.parse_args()
klasor = args.data_dir
if not all((klasor / f"batch{i}.dat").exists() for i in range(1, 11)):
    parser.error("data/ altında batch1.dat ... batch10.dat bulunmalı (veya --data-dir belirtin).")
proje = Path(__file__).resolve().parent
grafik_yolu = proje / "figures"
tablo_yolu = proje / "tables"
sonuc_yolu = proje / "results"
for yol in (grafik_yolu, tablo_yolu, sonuc_yolu):
    yol.mkdir(parents=True, exist_ok=True)

X_listesi, y_listesi, batch_listesi = [], [], []
for i in range(1, 11):
    X_part, y_part = load_svmlight_file(str(klasor / f"batch{i}.dat"), n_features=128)
    X_listesi.append(X_part.toarray())
    y_listesi.append(y_part)
    batch_listesi.append(np.full(len(y_part), i))
    print(f"Batch {i}: {X_part.shape[0]} örnek")
X = np.vstack(X_listesi)
y = np.concatenate(y_listesi)
batch = np.concatenate(batch_listesi)
print("Toplam örnek:", len(y), "Öznitelik:", X.shape[1])

# ---- Orijinal deney adımı 2 ----
batch_dagilimi = pd.Series(batch).value_counts().sort_index()
sinif_dagilimi = pd.Series(y.astype(int)).value_counts().sort_index()

# Grafik 1 - Batch dagilimi
plt.figure(figsize=(10, 5))
plt.bar(batch_dagilimi.index, batch_dagilimi.values)
plt.xlabel("Batch numarasi")
plt.ylabel("Ornek sayisi")
plt.xticks(range(1, 11))

for i, v in batch_dagilimi.items():
    plt.text(i, v + 40, str(v), ha="center")

plt.ylim(0, 4200)
plt.tight_layout()
plt.savefig(grafik_yolu / "01_Batch_Dagilimi.png", dpi=300)
plt.close()

# Grafik 2 - Sinif dagilimi
plt.figure(figsize=(8, 5))
plt.bar(sinif_dagilimi.index, sinif_dagilimi.values)
plt.xlabel("Gaz sinifi")
plt.ylabel("Ornek sayisi")
plt.xticks(range(1, 7))

for i, v in sinif_dagilimi.items():
    plt.text(i, v + 40, str(v), ha="center")

plt.ylim(0, 3500)
plt.tight_layout()
plt.savefig(grafik_yolu / "02_Sinif_Dagilimi.png", dpi=300)
plt.close()

# Tablolarin kaydedilmesi
batch_dagilimi.to_csv(tablo_yolu / "01_Batch_Dagilimi.csv", header=["Ornek sayisi"], index_label="Batch")
sinif_dagilimi.to_csv(tablo_yolu / "02_Sinif_Dagilimi.csv", header=["Ornek sayisi"], index_label="Sinif")

print("Grafikler ve tablolar basariyla kaydedildi.")
print("Grafik klasoru:", grafik_yolu)
print("Tablo klasoru:", tablo_yolu)


# ---- Orijinal deney adımı 3 ----
genel_istatistik = pd.DataFrame({
    "Ozellik": [
        "Toplam ornek sayisi",
        "Oznitelik sayisi",
        "Gaz sinifi sayisi",
        "Veri dosyasi sayisi",
        "Sensor sayisi",
        "Veri toplama suresi (yil)",
        "Eksik deger sayisi (NaN)",
        "Sonsuz deger sayisi",
        "Minimum oznitelik degeri",
        "Maksimum oznitelik degeri",
        "Ortalama",
        "Standart sapma"
    ],
    "Deger": [
        X.shape[0],
        X.shape[1],
        len(np.unique(y)),
        len(np.unique(batch)),
        16,
        3,
        np.isnan(X).sum(),
        np.isinf(X).sum(),
        np.min(X),
        np.max(X),
        np.mean(X),
        np.std(X)
    ]
})

genel_istatistik.to_csv(
    tablo_yolu / "T1_Genel_Istatistik.csv",
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig"
)

print(genel_istatistik.to_string(index=False))
print("T1 tablosu kaydedildi.")


# ---- Orijinal deney adımı 4 ----
oznitelik_tablosu = pd.DataFrame({
    "No": range(1, 9),
    "Öznitelik": [
        "ΔR", "|ΔR|",
        "EMAi 0.001", "EMAi 0.01", "EMAi 0.1",
        "EMAd 0.001", "EMAd 0.01", "EMAd 0.1"
    ],
    "Açıklama": [
        "Maksimum direnç değişimi",
        "Normalize edilmiş direnç değişimi",
        "Yükselen sensör tepkisi",
        "Yükselen sensör tepkisi",
        "Yükselen sensör tepkisi",
        "Azalan sensör tepkisi",
        "Azalan sensör tepkisi",
        "Azalan sensör tepkisi"
    ],
    "Sensör sayısı": [16] * 8
})

oznitelik_tablosu.to_csv(
    tablo_yolu / "T2_Oznitelik_Tanimlama.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print(oznitelik_tablosu.to_string(index=False))
print("T2 tablosu kaydedildi.")


# ---- Orijinal deney adımı 5 ----
# Inceleme icin secilen ilk 8 oznitelik
secili = range(8)

# Histogram
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, ax in zip(secili, axes.ravel()):
    ax.hist(X[:, i], bins=50, edgecolor="black")
    ax.set_title(f"Öznitelik {i+1}")
    ax.set_xlabel("Öznitelik değeri")
    ax.set_ylabel("Frekans")

plt.tight_layout()
plt.savefig(grafik_yolu / "G1_Histogram.png", dpi=300)
plt.close()

# Box plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, ax in zip(secili, axes.ravel()):
    ax.boxplot(X[:, i])
    ax.set_title(f"Öznitelik {i+1}")
    ax.set_ylabel("Öznitelik değeri")
    ax.set_xticks([])

plt.tight_layout()
plt.savefig(grafik_yolu / "G1_Boxplot.png", dpi=300)
plt.close()

print("Histogram ve Box Plot grafikleri kaydedildi.")


# ---- Orijinal deney adımı 6 ----
# Pearson korelasyon matrisinin hesaplanmasi
korelasyon = pd.DataFrame(
    X, columns=[f"F{i+1}" for i in range(X.shape[1])]
).corr(method="pearson")

# Korelasyon isi haritasi
fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(
    korelasyon,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    interpolation="nearest"
)

fig.colorbar(im, ax=ax, label="Korelasyon katsayısı")

etiketler = np.arange(0, 128, 8)

ax.set_xticks(etiketler)
ax.set_yticks(etiketler)
ax.set_xticklabels(etiketler + 1)
ax.set_yticklabels(etiketler + 1)

ax.set_xlabel("Öznitelik numarası")
ax.set_ylabel("Öznitelik numarası")

plt.tight_layout()
plt.savefig(
    grafik_yolu / "G2_Korelasyon_Isi_Haritasi.png",
    dpi=300
)
plt.close()

# Korelasyon matrisinin kaydedilmesi
korelasyon.to_csv(
    tablo_yolu / "Korelasyon_Matrisi.csv",
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

print("Korelasyon matris boyutu:", korelasyon.shape)
print("Korelasyon isi haritasi kaydedildi.")
print("Korelasyon matrisi kaydedildi.")


# ---- Orijinal deney adımı 7 ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tekrarlanabilirlik
SEED = 42

# Sinif etiketlerinin 0-5 araligina donusturulmesi
y_model = y.astype(int) - 1

# Verilerin egitim ve gecici gruba ayrilmasi
idx_train, idx_temp = train_test_split(
    np.arange(len(y_model)),
    test_size=0.30,
    random_state=SEED,
    stratify=y_model
)

# Dogrulama ve test gruplari
idx_val, idx_test = train_test_split(
    idx_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_model[idx_temp]
)

X_train = X[idx_train]
X_val = X[idx_val]
X_test = X[idx_test]

y_train = y_model[idx_train]
y_val = y_model[idx_val]
y_test = y_model[idx_test]

# Standardizasyon
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Veri sizintisi kontrolu
assert len(np.intersect1d(idx_train, idx_val)) == 0
assert len(np.intersect1d(idx_train, idx_test)) == 0
assert len(np.intersect1d(idx_val, idx_test)) == 0

print("VERI ON ISLEME SONUCLARI")
print("Egitim veri boyutu:", X_train.shape)
print("Dogrulama veri boyutu:", X_val.shape)
print("Test veri boyutu:", X_test.shape)
print("Sinif etiketleri:", np.unique(y_model))
print("Egitim verisi ortalamasi:", np.mean(X_train))
print("Egitim verisi standart sapmasi:", np.std(X_train))
print("Ortak ornek indeksi: Yok")
print("Veri on isleme tamamlandi.")


# ---- Orijinal deney adımı 8 ----
on_isleme_tablosu = pd.DataFrame({
    "Adım": range(1, 10),
    "İşlem": [
        "Verilerin okunması",
        "Verilerin birleştirilmesi",
        "Eksik ve sonsuz değer kontrolü",
        "Sınıf etiketlerinin düzenlenmesi",
        "Eğitim verisinin ayrılması",
        "Doğrulama verisinin ayrılması",
        "Test verisinin ayrılması",
        "Z-score standardizasyonu",
        "Ortak örnek indeksi kontrolü"
    ],
    "Sonuç": [
        "10 DAT dosyası",
        "13910 örnek ve 128 öznitelik",
        "NaN ve sonsuz değer bulunmadı",
        "1-6 etiketleri 0-5 olarak düzenlendi",
        "9737 örnek (%70)",
        "2086 örnek (yaklaşık %15)",
        "2087 örnek (yaklaşık %15)",
        "Eğitim ortalaması yaklaşık 0, standart sapması 1",
        "Veri gruplarında ortak indeks bulunmadı"
    ]
})

on_isleme_tablosu.to_csv(
    tablo_yolu / "T3_Veri_On_Isleme.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print(on_isleme_tablosu.to_string(index=False))
print("T3 tablosu kaydedildi.")


# ---- Orijinal deney adımı 9 ----
import torch
import torch.nn as nn

# Model 1: Normalizasyon katmani bulunmayan YSA
class StandardYSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.model(x)


# Model 2: Batch Normalization
class BNYSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.model(x)


# Model 3: Layer Normalization
class LNYSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.model(x)


torch.manual_seed(42)

models = {
    "Standart YSA": StandardYSA(),
    "BN-YSA": BNYSA(),
    "LN-YSA": LNYSA()
}

# Model yapilarinin kontrolu
for name, model in models.items():
    print("\\n", name)
    print(model)
    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print("Egitilebilir parametre sayisi:", parameters)

print("\\nUc model basariyla olusturuldu.")


# ---- Orijinal deney adımı 10 ----
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 9))

for ax, (name, model) in zip(axes, models.items()):

    katmanlar = ["Girdi (128)"]
    renkler = ["#DCEAF7"]

    for layer in model.model:
        if isinstance(layer, nn.Linear):
            katmanlar.append(
                f"Linear ({layer.out_features})"
            )
            renkler.append("#E5E5E5")

        elif isinstance(layer, nn.BatchNorm1d):
            katmanlar.append("Batch Normalization")
            renkler.append("#B5D5F5")

        elif isinstance(layer, nn.LayerNorm):
            katmanlar.append("Layer Normalization")
            renkler.append("#BCE4CD")

        elif isinstance(layer, nn.ReLU):
            katmanlar.append("ReLU")
            renkler.append("#FBE3B5")

    katmanlar.append("6 gaz sınıfı (skor)")
    renkler.append("#DCEAF7")

    konumlar = np.linspace(8.7, 1.2, len(katmanlar))

    for i, (katman, renk) in enumerate(
        zip(katmanlar, renkler)
    ):
        ax.text(
            0.5, konumlar[i], katman,
            ha="center", va="center",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor=renk,
                edgecolor="black"
            )
        )

        if i < len(katmanlar) - 1:
            ax.annotate(
                "",
                xy=(0.5, konumlar[i+1] + 0.25),
                xytext=(0.5, konumlar[i] - 0.25),
                arrowprops=dict(arrowstyle="->")
            )

    parametre = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    ax.text(
        0.5, 0.25,
        f"Eğitilebilir parametre: {parametre}",
        ha="center", fontsize=11
    )

    ax.set_title(name, fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    grafik_yolu / "G3_Model_Mimarisi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("G3 model mimarisi grafigi kaydedildi.")


# ---- Orijinal deney adımı 11 ----
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEEDS = [42, 123, 2026]

parametreler = [
    "Girdi sayısı",
    "Gizli katman sayısı",
    "Birinci gizli katman",
    "İkinci gizli katman",
    "Çıktı sayısı",
    "Normalizasyon",
    "Aktivasyon fonksiyonu",
    "Optimizasyon algoritması",
    "Kayıp fonksiyonu",
    "Öğrenme oranı",
    "Batch size",
    "Epoch",
    "Deney tekrarı",
    "Dropout",
    "Eğitilebilir parametre",
    "Epsilon",
    "BN momentum",
    "Weight decay",
    "Random seeds"
]

hiperparametre_tablosu = pd.DataFrame({
    "Parametre": parametreler
})

for name, model in models.items():

    norm = {
        "Standart YSA": "Yok",
        "BN-YSA": "Batch Normalization",
        "LN-YSA": "Layer Normalization"
    }[name]

    parametre_sayisi = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    hiperparametre_tablosu[name] = [
        128, 2, 128, 64, 6,
        norm,
        "ReLU",
        "Adam",
        "Cross Entropy",
        "0,001",
        BATCH_SIZE,
        EPOCHS,
        len(SEEDS),
        "Yok",
        parametre_sayisi,
        "Yok" if norm == "Yok" else "0,00001",
        "0,1" if name == "BN-YSA" else "Yok",
        0,
        str(SEEDS)
    ]

hiperparametre_tablosu.to_csv(
    tablo_yolu / "T4_Model_Hiperparametreleri.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print(hiperparametre_tablosu.to_string(index=False))
print("T4 tablosu kaydedildi.")


# ---- Orijinal deney adımı 12 ----
import platform
import sys
import os
import subprocess
import sklearn
import matplotlib

try:
    import psutil
    ram = f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    fiziksel_cekirdek = psutil.cpu_count(logical=False)
except ImportError:
    ram = "Manuel kontrol edilecek"
    fiziksel_cekirdek = "Manuel kontrol edilecek"

cuda = torch.cuda.is_available()

gpu = torch.cuda.get_device_name(0) if cuda else "CUDA ile kullanilamiyor"

ortam_tablosu = pd.DataFrame({
    "Özellik": [
        "İşletim sistemi",
        "İşlemci",
        "Fiziksel çekirdek sayısı",
        "Mantıksal çekirdek sayısı",
        "RAM",
        "GPU (CUDA)",
        "CUDA desteği",
        "Hesaplama cihazı",
        "Python sürümü",
        "PyTorch sürümü",
        "NumPy sürümü",
        "Pandas sürümü",
        "Scikit-learn sürümü",
        "Matplotlib sürümü"
    ],
    "Değer": [
        platform.platform(),
        platform.processor(),
        fiziksel_cekirdek,
        os.cpu_count(),
        ram,
        gpu,
        "Var" if cuda else "Yok",
        "CPU",
        platform.python_version(),
        torch.__version__,
        np.__version__,
        pd.__version__,
        sklearn.__version__,
        matplotlib.__version__
    ]
})

ortam_tablosu.to_csv(
    tablo_yolu / "T5_Hesaplama_Ortami.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print(ortam_tablosu.to_string(index=False))
print("T5 tablosu kaydedildi.")


# ---- Orijinal deney adımı 13 ----
guncel_bilgiler = {
    "İşletim sistemi": "Windows 11",
    "İşlemci": "Intel Core i5-13500H",
    "Fiziksel çekirdek sayısı": 12,
    "Mantıksal çekirdek sayısı": 16,
    "RAM": "16 GB",
    "GPU (CUDA)": "NVIDIA GeForce RTX (kullanılmadı)",
    "CUDA desteği": "Mevcut PyTorch sürümünde yok"
}

for ozellik, deger in guncel_bilgiler.items():
    ortam_tablosu.loc[
        ortam_tablosu["Özellik"] == ozellik,
        "Değer"
    ] = deger

ortam_tablosu.to_csv(
    tablo_yolu / "T5_Hesaplama_Ortami.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print(ortam_tablosu.to_string(index=False))
print("T5 tablosu guncellendi.")


# ---- Orijinal deney adımı 14 ----
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

device = torch.device("cpu")

print("EGITIM HAZIRLIK SONUCLARI")
print("Egitim ornek sayisi:", len(train_dataset))
print("Dogrulama ornek sayisi:", len(val_dataset))
print("Test ornek sayisi:", len(test_dataset))
print("Batch size:", BATCH_SIZE)
print("Hesaplama cihazi:", device)
print("Veriler egitime hazir.")


# ---- Orijinal deney adımı 15 ----
import time
import random
import copy

def seed_ayarla(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def model_egit(model, seed):

    seed_ayarla(seed)
    model = model.to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_state = None
    baslangic = time.perf_counter()

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        train_correct = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)

                val_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc*100:.2f}% | "
                f"Val Acc: {val_acc*100:.2f}%"
            )

    egitim_suresi = time.perf_counter() - baslangic

    model.load_state_dict(best_state)
    model.eval()

    return model, history, egitim_suresi

print("Model egitim fonksiyonu olusturuldu.")


# ---- Orijinal deney adımı 16 ----
model_tipleri = {
    "Standart_YSA": StandardYSA,
    "BN_YSA": BNYSA,
    "LN_YSA": LNYSA
}

seeds = [42, 123, 2026]

tum_gecmis = []
egitilen_modeller = {}
egitim_sureleri = []

for model_adi, model_sinifi in model_tipleri.items():

    for seed in seeds:

        print("=" * 60)
        print(f"Model: {model_adi} | Seed: {seed}")
        print("=" * 60)

        seed_ayarla(seed)
        model = model_sinifi()

        model, history, sure = model_egit(model, seed)

        egitilen_modeller[(model_adi, seed)] = model

        gecmis = pd.DataFrame(history)
        gecmis["Model"] = model_adi
        gecmis["Seed"] = seed

        tum_gecmis.append(gecmis)

        egitim_sureleri.append({
            "Model": model_adi,
            "Seed": seed,
            "Egitim suresi (s)": sure
        })

        gecmis.to_csv(
            sonuc_yolu / f"{model_adi}_{seed}_history.csv",
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig"
        )

        torch.save(
            model.state_dict(),
            sonuc_yolu / f"{model_adi}_{seed}.pth"
        )

        print(f"Egitim suresi: {sure:.2f} saniye")

pd.concat(tum_gecmis, ignore_index=True).to_csv(
    sonuc_yolu / "Tum_Egitim_Sonuclari.csv",
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

pd.DataFrame(egitim_sureleri).to_csv(
    sonuc_yolu / "Egitim_Sureleri.csv",
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

print("=" * 60)
print("9 deney basariyla tamamlandi.")
print("Tum egitim sonuclari kaydedildi.")


# ---- Orijinal deney adımı 17 ----
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_sonuclari = []
tahmin_sonuclari = []

criterion = nn.CrossEntropyLoss()

for (model_adi, seed), model in egitilen_modeller.items():

    model.eval()

    gercek = []
    tahmin = []
    test_loss = 0

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)

            gercek.extend(labels.cpu().numpy())
            tahmin.extend(outputs.argmax(dim=1).cpu().numpy())

    test_loss /= len(test_dataset)

    accuracy = accuracy_score(gercek, tahmin)

    precision = precision_score(
        gercek, tahmin, average="macro", zero_division=0
    )

    recall = recall_score(
        gercek, tahmin, average="macro", zero_division=0
    )

    f1 = f1_score(
        gercek, tahmin, average="macro", zero_division=0
    )

    test_sonuclari.append({
        "Model": model_adi,
        "Seed": seed,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Test Loss": test_loss
    })

    for i, (g, t) in enumerate(zip(gercek, tahmin)):

        tahmin_sonuclari.append({
            "Model": model_adi,
            "Seed": seed,
            "Test ornek no": i + 1,
            "Gercek sinif": g + 1,
            "Tahmin edilen sinif": t + 1
        })

test_df = pd.DataFrame(test_sonuclari)
tahmin_df = pd.DataFrame(tahmin_sonuclari)

test_df.to_csv(
    sonuc_yolu / "Test_Sonuclari.csv",
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

tahmin_df.to_csv(
    sonuc_yolu / "Test_Tahminleri.csv",
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

print("MODEL TEST SONUCLARI")
print(test_df.round(4).to_string(index=False))
print("9 modelin test islemi tamamlandi.")


# ---- Orijinal deney adımı 18 ----
metrikler = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "Test Loss"
]

t6_veri = test_df.merge(
    pd.DataFrame(egitim_sureleri),
    on=["Model", "Seed"]
)

t6_satirlar = []

for model_adi in ["Standart_YSA", "BN_YSA", "LN_YSA"]:

    grup = t6_veri[t6_veri["Model"] == model_adi]

    satir = {"Model": model_adi}

    for metrik in metrikler:

        ortalama = grup[metrik].mean()
        std = grup[metrik].std(ddof=1)

        if metrik != "Test Loss":
            ortalama *= 100
            std *= 100

        satir[metrik] = f"{ortalama:.4f} ± {std:.4f}"

    sure_ort = grup["Egitim suresi (s)"].mean()
    sure_std = grup["Egitim suresi (s)"].std(ddof=1)

    satir["Egitim suresi (s)"] = (
        f"{sure_ort:.2f} ± {sure_std:.2f}"
    )

    t6_satirlar.append(satir)

t6_tablosu = pd.DataFrame(t6_satirlar)

t6_tablosu = t6_tablosu.rename(columns={
    "Accuracy": "Accuracy (%)",
    "Precision": "Precision (%)",
    "Recall": "Recall (%)",
    "F1-Score": "F1-Score (%)"
})

t6_tablosu.to_csv(
    tablo_yolu / "T6_Ana_Performans_Metrikleri.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print("T6 - ANA PERFORMANS METRIKLERI")
print(t6_tablosu.to_string(index=False))
print("T6 tablosu basariyla kaydedildi.")


# ---- Orijinal deney adımı 19 ----
t7_ortalama = test_df.groupby("Model").mean(numeric_only=True)

referans_acc = t7_ortalama.loc["Standart_YSA", "Accuracy"]
referans_f1 = t7_ortalama.loc["Standart_YSA", "F1-Score"]
referans_loss = t7_ortalama.loc["Standart_YSA", "Test Loss"]

t7_satirlar = []

for model_adi in ["Standart_YSA", "BN_YSA", "LN_YSA"]:

    acc = t7_ortalama.loc[model_adi, "Accuracy"]
    f1 = t7_ortalama.loc[model_adi, "F1-Score"]
    loss = t7_ortalama.loc[model_adi, "Test Loss"]

    t7_satirlar.append({
        "Model": model_adi,
        "BN": "Var" if model_adi == "BN_YSA" else "Yok",
        "LN": "Var" if model_adi == "LN_YSA" else "Yok",
        "Accuracy (%)": acc * 100,
        "F1-Score (%)": f1 * 100,
        "Test Loss": loss,
        "Accuracy farki (yuzde puan)": (acc - referans_acc) * 100,
        "F1 farki (yuzde puan)": (f1 - referans_f1) * 100,
        "Test Loss farki": loss - referans_loss
    })

t7_tablosu = pd.DataFrame(t7_satirlar)

t7_tablosu.round(4).to_csv(
    tablo_yolu / "T7_Ablasyon_Calismasi.csv",
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

print("T7 - ABLASYON CALISMASI")
print(t7_tablosu.round(4).to_string(index=False))
print("T7 tablosu basariyla kaydedildi.")


# ---- Orijinal deney adımı 20 ----
import pandas as pd
import matplotlib.pyplot as plt

egitim_df = pd.read_csv(
    sonuc_yolu / "Tum_Egitim_Sonuclari.csv",
    sep=";",
    decimal=","
)

modeller = ["Standart_YSA", "BN_YSA", "LN_YSA"]
basliklar = ["Standart YSA", "BN-YSA", "LN-YSA"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, model_adi, baslik in zip(axes, modeller, basliklar):

    grup = egitim_df[egitim_df["Model"] == model_adi]

    ortalama = grup.groupby("epoch")[
        ["train_loss", "val_loss"]
    ].mean()

    ax.plot(
        ortalama.index,
        ortalama["train_loss"],
        label="Egitim kaybi",
        color="blue",
        linewidth=1.5
    )

    ax.plot(
        ortalama.index,
        ortalama["val_loss"],
        label="Dogrulama kaybi",
        color="red",
        linewidth=1.5
    )

    ax.set_title(baslik)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kayip (Loss)")
    ax.set_xlim(1, 100)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    grafik_yolu / "G4_Egitim_Dogrulama_Kayip_Egrisi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("G4 kayip egrisi basariyla kaydedildi.")


# ---- Orijinal deney adımı 21 ----
import pandas as pd
import matplotlib.pyplot as plt

egitim_df = pd.read_csv(
    sonuc_yolu / "Tum_Egitim_Sonuclari.csv",
    sep=";",
    decimal=","
)

modeller = ["Standart_YSA", "BN_YSA", "LN_YSA"]
basliklar = ["Standart YSA", "BN-YSA", "LN-YSA"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, model_adi, baslik in zip(axes, modeller, basliklar):

    grup = egitim_df[egitim_df["Model"] == model_adi]

    ortalama = grup.groupby("epoch")[
        ["train_acc", "val_acc"]
    ].mean()

    carpan = 100 if ortalama.max().max() <= 1 else 1

    ax.plot(
        ortalama.index,
        ortalama["train_acc"] * carpan,
        label="Eğitim doğruluğu",
        color="blue",
        linewidth=1.5
    )

    ax.plot(
        ortalama.index,
        ortalama["val_acc"] * carpan,
        label="Doğrulama doğruluğu",
        color="red",
        linewidth=1.5
    )

    ax.set_title(baslik)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Doğruluk (%)")
    ax.set_xlim(1, 100)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    grafik_yolu / "G5_Egitim_Dogrulama_Dogruluk_Egrisi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("G5 dogruluk egrisi basariyla kaydedildi.")


# ---- Orijinal deney adımı 22 ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_df = pd.read_csv(
    sonuc_yolu / "Test_Sonuclari.csv",
    sep=";",
    decimal=","
)

modeller = ["Standart_YSA", "BN_YSA", "LN_YSA"]
metrikler = ["Accuracy", "Precision", "Recall", "F1-Score"]

ortalama = test_df.groupby("Model")[metrikler].mean() * 100
std = test_df.groupby("Model")[metrikler].std() * 100

x = np.arange(len(metrikler))
genislik = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

renkler = ["#7F8C8D", "#3498DB", "#2ECC71"]
etiketler = ["Standart YSA", "BN-YSA", "LN-YSA"]

for i, model_adi in enumerate(modeller):

    degerler = ortalama.loc[model_adi].values
    sapmalar = std.loc[model_adi].values

    cubuklar = ax.bar(
        x + (i - 1) * genislik,
        degerler,
        genislik,
        yerr=sapmalar,
        capsize=4,
        label=etiketler[i],
        color=renkler[i],
        edgecolor="black",
        linewidth=0.5
    )

    for cubuk, deger, sapma in zip(
        cubuklar, degerler, sapmalar
    ):
        ax.text(
            cubuk.get_x() + cubuk.get_width() / 2,
            deger + sapma + 0.04,
            f"{deger:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

ax.set_xticks(x)
ax.set_xticklabels(metrikler)
ax.set_ylabel("Performans (%)")
ax.set_xlabel("Performans metrikleri")
ax.set_ylim(97.5, 101)
ax.legend(loc="upper center", ncol=3)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    grafik_yolu / "G6_Karsilastirmali_Yontem_Analizi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("G6 karsilastirmali yontem analizi kaydedildi.")


# ---- Orijinal deney adımı 23 ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

tahmin_df = pd.read_csv(
    sonuc_yolu / "Test_Tahminleri.csv",
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

modeller = ["Standart_YSA", "BN_YSA", "LN_YSA"]
basliklar = ["Standart YSA", "BN-YSA", "LN-YSA"]

matrisler = []

for model_adi in modeller:
    grup = tahmin_df[tahmin_df["Model"] == model_adi]

    cm = confusion_matrix(
        grup["Gercek sinif"],
        grup["Tahmin edilen sinif"],
        labels=[1, 2, 3, 4, 5, 6]
    )

    matrisler.append(cm)

    print(model_adi)
    print(cm)
    print("Toplam tahmin:", cm.sum())
    print("Dogru tahmin:", cm.trace())
    print("Yanlis tahmin:", cm.sum() - cm.trace())
    print()

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

vmax = max(cm.max() for cm in matrisler)

for ax, cm, baslik in zip(axes, matrisler, basliklar):

    im = ax.imshow(
        cm,
        cmap="Blues",
        vmin=0,
        vmax=vmax
    )

    ax.set_title(baslik)
    ax.set_xlabel("Tahmin edilen sinif")
    ax.set_ylabel("Gercek sinif")

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))

    ax.set_xticklabels(range(1, 7))
    ax.set_yticklabels(range(1, 7))

    for i in range(6):
        for j in range(6):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > vmax / 2 else "black",
                fontsize=10
            )

fig.subplots_adjust(
    left=0.06,
    right=0.90,
    bottom=0.12,
    top=0.90,
    wspace=0.30
)

cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
fig.colorbar(im, cax=cax, label="Tahmin sayisi")

plt.savefig(
    grafik_yolu / "G7_Confusion_Matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("G7 confusion matrix basariyla kaydedildi.")


# ---- Orijinal deney adımı 24 ----
t10_satirlar = []

for model_adi in ["Standart_YSA", "BN_YSA", "LN_YSA"]:

    grup = tahmin_df[tahmin_df["Model"] == model_adi]

    for sinif in range(1, 7):

        sinif_grubu = grup[grup["Gercek sinif"] == sinif]

        toplam = len(sinif_grubu)

        hatalar = sinif_grubu[
            sinif_grubu["Gercek sinif"] !=
            sinif_grubu["Tahmin edilen sinif"]
        ]

        yanlis = len(hatalar)
        dogru = toplam - yanlis

        hata_orani = yanlis / toplam * 100 if toplam else 0

        en_cok = (
            hatalar["Tahmin edilen sinif"].mode()
        )

        karistirilan = (
            ", ".join(str(int(x)) for x in en_cok)
            if len(en_cok) > 0 else "Yok"
        )

        t10_satirlar.append({
            "Model": model_adi,
            "Gaz sinifi": sinif,
            "Toplam tahmin": toplam,
            "Dogru tahmin": dogru,
            "Yanlis tahmin": yanlis,
            "Hata orani (%)": hata_orani,
            "En cok karistirilan sinif": karistirilan
        })

t10_tablosu = pd.DataFrame(t10_satirlar)

t10_tablosu.round(4).to_csv(
    tablo_yolu / "T10_Hata_Analizi.csv",
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

print("T10 - HATA ANALIZI")
print(t10_tablosu.round(4).to_string(index=False))
print("T10 tablosu basariyla kaydedildi.")
