import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Sayfa ayarları
st.set_page_config(page_title="Yaş Tahmini", layout="centered")

IMG_SIZE = 224
NUM_CLASSES = 80

# MAE metric (model yüklerken gerekli)
def mae_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ages = tf.range(1, NUM_CLASSES + 1, dtype=tf.float32)
    true_ages = tf.reduce_sum(y_true * ages, axis=1)
    pred_ages = tf.reduce_sum(y_pred * ages, axis=1)
    return tf.reduce_mean(tf.abs(pred_ages - true_ages))

# Model yükleme (cache ile)
@st.cache_resource
def load_age_model():
    return load_model("age_dex_resnet50_finetuned_final.h5", 
                     custom_objects={'mae_metric': mae_metric})

# Tahmin fonksiyonu (test_foto.py ile aynı preprocessing)
def predict_age(uploaded_file):
    model = load_age_model()
    
    # test_foto.py ile aynı şekilde: Keras image.load_img() kullan
    img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # test_foto.py ile aynı tahmin
    probs = model.predict(img_array, verbose=0)
    predicted_age = np.sum(probs * np.arange(1, NUM_CLASSES + 1), axis=1)[0]
    return predicted_age

# UI
st.title("Yaş Tahmini")
st.markdown("---")

# Fotoğraf yükleme
uploaded_file = st.file_uploader("Fotoğrafınızı sürükleyip bırakın", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Resmi göster (PIL ile)
    img_display = Image.open(uploaded_file)
    st.image(img_display, width=300)
    
    # Tahmin yap (test_foto.py ile aynı preprocessing)
    with st.spinner("Tahmin yapılıyor..."):
        # Dosyayı başa sar (Keras için)
        uploaded_file.seek(0)
        age = predict_age(uploaded_file)
    
    # Sonuç
    st.markdown("---")
    st.markdown(f"### Tahmin Edilen Yaş: **{age:.1f}** yaş")

