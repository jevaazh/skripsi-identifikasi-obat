import streamlit as st
import numpy as np
import cv2
import pickle
import re
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

st.set_page_config(page_title="Identifikasi Sisa Obat", page_icon="💊")

# ===== KONFIGURASI FINAL (dari Colab, jangan diubah) =====
COLOR_RANGES = {
    'Biru Tua': {
        'key': 'biru',
        'lower': np.array([95, 130, 50]),
        'upper': np.array([130, 255, 255]),
        'threshold': 1.123
    },
    'Hijau Stabilo': {
        'key': 'hijau',
        'lower': np.array([30, 135, 75]),
        'upper': np.array([95, 255, 255]),
        'threshold': 1.334
    }
}

# ===== FUNGSI EKSTRAKSI FITUR (identik dengan Colab) =====
def segment_object(img_bgr, lower_bg, upper_bg):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)
    mask_obj = cv2.bitwise_not(mask_bg)
    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)
    mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel_close)
    mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel_open)
    return mask_obj

def extract_shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
    if not valid_contours:
        return None
    area = sum(cv2.contourArea(c) for c in valid_contours)
    perimeter = sum(cv2.arcLength(c, True) for c in valid_contours)
    if perimeter == 0 or area < 80:
        return None
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    aspect_ratio = float(w) / h if h != 0 else 0
    return area, perimeter, circularity, aspect_ratio

def extract_color_features(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_vals = cv2.mean(hsv, mask=mask)
    return mean_vals[0], mean_vals[1], mean_vals[2]

def extract_texture_features(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    glcm = graycomatrix(gray_masked, distances=[1],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return contrast, correlation, energy, homogeneity

@st.cache_resource
def load_model(warna_key):
    with open(f'model/knn_{warna_key}.pkl', 'rb') as f:
        knn = pickle.load(f)
    with open(f'model/scaler_{warna_key}.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return knn, scaler

# ===== UI =====
st.title("💊 Identifikasi Sisa Obat")
st.caption("Purwarupa Stationary Scanner — Skripsi KNN + Feature Engineering")

warna_pilihan = st.selectbox("Pilih warna alas yang digunakan:", list(COLOR_RANGES.keys()))
cfg = COLOR_RANGES[warna_pilihan]

uploaded_file = st.file_uploader("Unggah citra sisa obat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Citra masukan", use_container_width=True)

    img_array = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (600, 800))

    mask = segment_object(img_bgr, cfg['lower'], cfg['upper'])
    shape = extract_shape_features(mask)

    if shape is None:
        st.error("Objek tidak terdeteksi pada citra. Pastikan objek terlihat jelas dan alas sesuai warna yang dipilih.")
    else:
        color = extract_color_features(img_bgr, mask)
        texture = extract_texture_features(img_bgr, mask)
        fitur = np.array(list(shape) + list(color) + list(texture)).reshape(1, -1)

        knn, scaler = load_model(cfg['key'])
        fitur_scaled = scaler.transform(fitur)

        distance, _ = knn.kneighbors(fitur_scaled, n_neighbors=1)
        distance = distance[0][0]

        if distance > cfg['threshold']:
            st.warning("⚠️ Obat tidak dikenali oleh sistem. Objek berada di luar cakupan 20 kelas obat yang dilatih.")
        else:
            pred_subkelas = knn.predict(fitur_scaled)[0]
            pred_obat = re.sub(r'_\d$', '', pred_subkelas).replace('_', ' ')
            st.success(f"**Obat teridentifikasi: {pred_obat}**")
            st.caption(f"Jarak ke tetangga terdekat: {distance:.3f} (ambang batas: {cfg['threshold']:.3f})")