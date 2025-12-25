import os
import cv2
import shutil
import random
from tqdm import tqdm

# ==============================
# AYARLAR
# ==============================
SOURCE_DIRS = [
    "dataset/crop_part1",
    "dataset/UTKFace"
]

OUTPUT_DIR = "dataset_clean"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_AGE = 1
MAX_AGE = 80
MIN_IMG_SIZE = 80
SEED = 42

random.seed(SEED)

# ==============================
# KLASÖR OLUŞTUR
# ==============================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ==============================
# DOSYA ADI PARSE
# ==============================
def parse_filename(fname):
    try:
        parts = fname.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        return age, gender, race
    except:
        return None

# ==============================
# TÜM GEÇERLİ DOSYALARI TOPLA
# ==============================
valid_samples = []

for src_dir in SOURCE_DIRS:
    if not os.path.exists(src_dir):
        print(f"Uyarı: {src_dir} klasörü bulunamadı, atlanıyor...")
        continue
    
    for fname in tqdm(os.listdir(src_dir), desc=f"Scanning {src_dir}"):
        if not fname.lower().endswith(".jpg"):
            continue
        
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        
        age, gender, race = parsed
        
        if age < MIN_AGE or age > MAX_AGE:
            continue
        
        path = os.path.join(src_dir, fname)
        img = cv2.imread(path)
        
        if img is None:
            continue
        
        h, w, _ = img.shape
        
        if h < MIN_IMG_SIZE or w < MIN_IMG_SIZE:
            continue
        
        valid_samples.append((path, fname))

print(f"\nToplam temiz veri sayısı: {len(valid_samples)}")

# ==============================
# SHUFFLE + SPLIT
# ==============================
random.shuffle(valid_samples)
n_total = len(valid_samples)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_samples = valid_samples[:n_train]
val_samples = valid_samples[n_train:n_train + n_val]
test_samples = valid_samples[n_train + n_val:]

# ==============================
# KOPYALA
# ==============================
def copy_samples(samples, split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    for src_path, fname in tqdm(samples, desc=f"Copying {split_name}"):
        dst_path = os.path.join(split_dir, fname)
        shutil.copy2(src_path, dst_path)

copy_samples(train_samples, "train")
copy_samples(val_samples, "val")
copy_samples(test_samples, "test")

print("\nDataset başarıyla hazırlandı!")
print(f"Train: {len(train_samples)}")
print(f"Val:   {len(val_samples)}")
print(f"Test:  {len(test_samples)}")

