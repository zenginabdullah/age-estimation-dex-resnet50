import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
# ImageDataGenerator kaldırıldı - tf.image kullanılıyor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# ==============================
# AYARLAR
# ==============================
DATASET_DIR = "dataset_clean"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 80   # age 1–80
LEARNING_RATE = 1e-4

# ==============================
# YAŞ PARSE FONKSİYONU
# ==============================
def age_from_filename(fname):
    # age_23_000123.jpg -> 23
    try:
        return int(fname.split("_")[1]) - 1  # 0-based
    except:
        return None

# ==============================
# DATA GENERATOR (tf.image ile temiz augmentation)
# ==============================
def generator(split, augment=False):
    files = [f for f in os.listdir(os.path.join(DATASET_DIR, split)) 
             if f.lower().endswith('.jpg')]
    
    while True:
        np.random.shuffle(files)
        
        for i in range(0, len(files), BATCH_SIZE):
            batch_files = files[i:i+BATCH_SIZE]
            
            images = []
            labels = []
            
            for fname in batch_files:
                path = os.path.join(DATASET_DIR, split, fname)
                
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        path, target_size=(IMG_SIZE, IMG_SIZE)
                    )
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    
                    # tf.image ile augmentation (GPU'da daha hızlı, deterministik)
                    if augment:
                        # Horizontal flip
                        if np.random.random() > 0.5:
                            img = tf.image.flip_left_right(img)
                        
                        # Brightness
                        img = tf.image.random_brightness(img, max_delta=0.2)
                        
                        # Contrast
                        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
                        
                        # Saturation (renk doygunluğu)
                        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
                    
                    # Normalize (her zaman aynı şekilde)
                    img = img / 255.0
                    
                    age = age_from_filename(fname)
                    if age is None:
                        continue
                    
                    label = tf.keras.utils.to_categorical(age, NUM_CLASSES)
                    
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Hata: {fname} - {e}")
                    continue
            
            if len(images) > 0:
                images = np.array(images)
                labels = np.array(labels)
                
                yield images, labels

# ==============================
# MAE METRIC (DEX Expected Value ile)
# ==============================
def mae_metric(y_true, y_pred):
    """
    DEX yöntemine göre MAE metriği
    y_true: categorical labels (one-hot encoded)
    y_pred: softmax probabilities
    """
    # Dtype uyumluluğu için float32'ye cast et
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Age değerleri: 1, 2, 3, ..., NUM_CLASSES
    ages = tf.range(1, NUM_CLASSES + 1, dtype=tf.float32)
    
    # True age'leri DEX expected value mantığıyla hesapla (one-hot olmayan label'lar için de çalışır)
    true_ages = tf.reduce_sum(y_true * ages, axis=1)
    
    # Predicted age'leri DEX expected value ile hesapla
    pred_ages = tf.reduce_sum(y_pred * ages, axis=1)
    
    # MAE hesapla
    mae = tf.reduce_mean(tf.abs(pred_ages - true_ages))
    return mae

# ==============================
# MODEL (DEX)
# ==============================
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Önce frozen

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)  # 256'dan 512'ye artırıldı
x = layers.BatchNormalization()(x)  # BatchNorm eklendi
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)  # İkinci Dense layer eklendi
x = layers.Dropout(0.3)(x)
output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=[mae_metric]  # DEX makalesine göre MAE kullanılıyor
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
callbacks = [
    EarlyStopping(
        monitor='val_mae_metric',  # MAE'yi monitor et (düşük olmalı)
        mode='min',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_age_model.h5',
        monitor='val_mae_metric',  # En iyi MAE'ye sahip modeli kaydet
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_mae_metric',  # MAE'ye göre learning rate ayarla
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# ==============================
# EĞİTİM
# ==============================
train_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "train")) 
                if f.lower().endswith('.jpg')]
val_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "val")) 
              if f.lower().endswith('.jpg')]

train_steps = len(train_files) // BATCH_SIZE
val_steps = len(val_files) // BATCH_SIZE

print(f"Train örnekleri: {len(train_files)}")
print(f"Val örnekleri: {len(val_files)}")
print(f"Train steps per epoch: {train_steps}")
print(f"Val steps: {val_steps}")

history = model.fit(
    generator("train", augment=True),
    steps_per_epoch=train_steps,
    validation_data=generator("val", augment=False),
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ==============================
# FINE-TUNING (Opsiyonel - daha iyi sonuç için)
# ==============================
print("\nFine-tuning başlatılıyor...")
base_model.trainable = True

# Son birkaç layer'ı freeze et
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(LEARNING_RATE / 10),  # Daha düşük learning rate
    loss="categorical_crossentropy",
    metrics=[mae_metric]  # DEX makalesine göre MAE kullanılıyor
)

history_finetune = model.fit(
    generator("train", augment=True),
    steps_per_epoch=train_steps,
    validation_data=generator("val", augment=False),
    validation_steps=val_steps,
    epochs=10,  # Fine-tuning için daha az epoch
    callbacks=callbacks,
    verbose=1
)

# ==============================
# EXPECTED VALUE (DEX) - Makaleye göre
# ==============================
def dex_predict(model, images):
    probs = model.predict(images, verbose=0)
    ages = np.arange(1, NUM_CLASSES + 1)
    return np.sum(probs * ages, axis=1)

# ==============================
# TEST MAE
# ==============================
def evaluate_test():
    errors = []
    test_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "test")) 
                   if f.lower().endswith('.jpg')]
    
    print(f"\nTest setinde {len(test_files)} örnek değerlendiriliyor...")
    
    for fname in test_files:
        path = os.path.join(DATASET_DIR, "test", fname)
        
        try:
            img = tf.keras.preprocessing.image.load_img(
                path, target_size=(IMG_SIZE, IMG_SIZE)
            )
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            
            age = age_from_filename(fname)
            if age is None:
                continue
            
            true_age = age + 1
            pred_age = dex_predict(model, img)[0]
            
            errors.append(abs(pred_age - true_age))
        except Exception as e:
            print(f"Hata: {fname} - {e}")
            continue
    
    mae = np.mean(errors)
    print(f"\n{'='*50}")
    print(f"TEST MAE: {mae:.2f} yaş")
    print(f"Ortalama hata: {mae:.2f} ± {np.std(errors):.2f} yaş")
    print(f"Medyan hata: {np.median(errors):.2f} yaş")
    print(f"{'='*50}")

evaluate_test()

# Model kaydet
model.save("age_dex_resnet50.h5")
print("\nModel kaydedildi: age_dex_resnet50.h5")
