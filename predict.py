import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

# ==============================
# AYARLAR
# ==============================
IMG_SIZE = 224
NUM_CLASSES = 80
MODEL_PATH = "age_dex_resnet50_finetuned_final.h5"  # En iyi modeli kullan

# ==============================
# MAE METRIC (Model yüklerken gerekli)
# ==============================
def mae_metric(y_true, y_pred):
    """DEX yöntemine göre MAE metriği"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ages = tf.range(1, NUM_CLASSES + 1, dtype=tf.float32)
    true_ages = tf.reduce_sum(y_true * ages, axis=1)
    pred_ages = tf.reduce_sum(y_pred * ages, axis=1)
    mae = tf.reduce_mean(tf.abs(pred_ages - true_ages))
    return mae

# ==============================
# YAŞ PARSE FONKSİYONU
# ==============================
def age_from_filename(fname):
    """
    Dosya isminden gerçek yaşı parse et
    age_23_000123.jpg -> 23
    """
    try:
        # age_X formatından yaşı al
        parts = os.path.basename(fname).split("_")
        if len(parts) >= 2 and parts[0] == "age":
            return int(parts[1])
        return None
    except:
        return None

# ==============================
# DEX PREDICT (Expected Value)
# ==============================
def dex_predict(model, img_array):
    """
    DEX yöntemiyle yaş tahmini
    Args:
        model: Yüklenmiş Keras modeli
        img_array: Preprocess edilmiş görüntü array'i (1, 224, 224, 3)
    Returns:
        Tahmin edilen yaş (float)
    """
    probs = model.predict(img_array, verbose=0)
    ages = np.arange(1, NUM_CLASSES + 1)
    predicted_age = np.sum(probs * ages, axis=1)[0]
    return predicted_age

# ==============================
# RESİM YÜKLEME VE PREPROCESS
# ==============================
def preprocess_image(img_path):
    """
    Resmi yükleyip model için hazırla
    Args:
        img_path: Resim dosyasının yolu
    Returns:
        Preprocess edilmiş görüntü array'i
    """
    # Resmi yükle ve resize et
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    
    # Normalize et (0-1 aralığına)
    img_array = img_array / 255.0
    
    # Batch dimension ekle (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ==============================
# KLASÖR TAHMİNİ (Gerçek yaş ile karşılaştırma)
# ==============================
def predict_folder(model, folder_path):
    """
    Bir klasördeki tüm resimler için tahmin yap ve gerçek yaşlarla karşılaştır
    """
    if not os.path.exists(folder_path):
        print(f"Hata: {folder_path} bulunamadı!")
        return
    
    # Desteklenen resim formatları
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Tüm resimleri bul
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if len(image_files) == 0:
        print(f"Klasörde resim bulunamadı: {folder_path}")
        return
    
    # Dosya isimlerine göre sırala
    image_files.sort()
    
    print(f"\n{len(image_files)} resim bulundu. Tahmin yapılıyor...\n")
    print(f"{'='*70}")
    print(f"{'Dosya':<40} {'Gerçek Yaş':<15} {'Tahmin Edilen Yaş':<20} {'Hata':<10}")
    print(f"{'='*70}")
    
    results = []
    errors = []
    
    for i, img_path in enumerate(image_files, 1):
        try:
            # Gerçek yaşı parse et
            true_age = age_from_filename(img_path)
            
            # Tahmin yap
            img_array = preprocess_image(img_path)
            predicted_age = dex_predict(model, img_array)
            
            # Hata hesapla
            if true_age is not None:
                error = abs(predicted_age - true_age)
                errors.append(error)
            else:
                error = None
            
            # Sonuçları kaydet
            filename = os.path.basename(img_path)
            results.append({
                'filename': filename,
                'true_age': true_age,
                'predicted_age': predicted_age,
                'error': error
            })
            
            # Çıktı
            if true_age is not None:
                print(f"{filename:<40} {true_age:<15.0f} {predicted_age:<20.1f} {error:<10.1f}")
            else:
                print(f"{filename:<40} {'Bilinmiyor':<15} {predicted_age:<20.1f} {'-':<10}")
                
        except Exception as e:
            print(f"Hata ({os.path.basename(img_path)}): {e}")
    
    # Özet istatistikler
    if results and errors:
        print(f"{'='*70}\n")
        print("ÖZET İSTATİSTİKLER")
        print(f"{'='*70}")
        print(f"Toplam resim: {len(results)}")
        print(f"Gerçek yaş bilgisi olan: {len(errors)}")
        
        # MAE (Mean Absolute Error)
        mae = np.mean(errors)
        print(f"\nOrtalama Mutlak Hata (MAE): {mae:.2f} yaş")
        print(f"Standart Sapma: {np.std(errors):.2f} yaş")
        print(f"Medyan Hata: {np.median(errors):.2f} yaş")
        print(f"Min Hata: {np.min(errors):.1f} yaş")
        print(f"Max Hata: {np.max(errors):.1f} yaş")
        
        # Doğruluk yüzdeleri (farklı toleranslarla)
        print(f"\n{'='*70}")
        print("TAHMIN DOĞRULUĞU (Tolerans Bazlı)")
        print(f"{'='*70}")
        
        # Errors listesini temizle (None veya geçersiz değerleri filtrele)
        valid_errors = [e for e in errors if e is not None and not np.isnan(e) and e >= 0]
        total_valid = len(valid_errors)
        
        if total_valid == 0:
            print("Geçerli hata verisi bulunamadı!")
        else:
            for tolerance in [1, 2, 3, 5]:
                correct = sum(1 for e in valid_errors if e <= tolerance)
                accuracy = (correct / total_valid) * 100
                print(f"±{tolerance} yaş toleransı: {correct}/{total_valid} doğru ({accuracy:.1f}%)")
            
            # Tam eşleşme (float karşılaştırma için küçük tolerans)
            exact_match = sum(1 for e in valid_errors if abs(e) < 0.1)  # 0.1 yaş toleransı ile tam eşleşme
            exact_accuracy = (exact_match / total_valid) * 100
            print(f"Tam eşleşme (<0.1 yaş hata): {exact_match}/{total_valid} ({exact_accuracy:.1f}%)")
        
        print(f"{'='*70}")

# ==============================
# ANA FONKSİYON
# ==============================
def main():
    parser = argparse.ArgumentParser(description='Yaş Tahmini - DEX Modeli (Klasör Tarama)')
    parser.add_argument('folder', type=str, help='Klasör yolu (içindeki tüm resimler taranacak)')
    parser.add_argument('--model', type=str, default=MODEL_PATH, 
                        help=f'Model dosyası yolu (varsayılan: {MODEL_PATH})')
    
    args = parser.parse_args()
    
    # Model yükle
    print(f"Model yükleniyor: {args.model}")
    try:
        model = load_model(args.model, custom_objects={'mae_metric': mae_metric})
        print("Model başarıyla yüklendi!\n")
    except Exception as e:
        print(f"Hata: Model yüklenemedi - {e}")
        return
    
    # Klasör kontrolü
    if not os.path.isdir(args.folder):
        print(f"Hata: {args.folder} geçerli bir klasör değil!")
        return
    
    # Klasörü tara
    predict_folder(model, args.folder)

if __name__ == "__main__":
    main()