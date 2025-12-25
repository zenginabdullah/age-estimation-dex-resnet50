import os

BASE_DIR = "dataset_clean"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)
    if not os.path.exists(split_dir):
        print(f"Uyarı: {split_dir} bulunamadı!")
        continue
    
    files = sorted(os.listdir(split_dir))

    counter = 0
    for fname in files:
        if not fname.lower().endswith(".jpg"):
            continue

        # Eski format: 23_1_0_000123.jpg veya yeni format: age_23_000123.jpg
        try:
            if fname.startswith("age_"):
                # Zaten yeni formatta
                continue
            else:
                # Eski format: ilk kısım yaş
                age = int(fname.split("_")[0])
        except:
            continue

        new_name = f"age_{age}_{counter:06d}.jpg"

        src = os.path.join(split_dir, fname)
        dst = os.path.join(split_dir, new_name)

        try:
            os.rename(src, dst)
            counter += 1
        except Exception as e:
            print(f"Hata: {fname} -> {new_name}: {e}")

    print(f"{split} klasörü yeniden adlandırıldı ({counter} dosya).")