import streamlit as st
import numpy as np
import cv2
import pickle
import re
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# Setup UI halaman
st.set_page_config(page_title="Identifikasi Sisa Obat", page_icon="💊", layout="centered")

# ===== KONFIGURASI FINAL =====
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

# ===== KAMUS OBAT & PABRIKAN BERDASARKAN KATEGORI PENYAKIT =====
KATEGORI_OBAT = {
    "Hipertensi (Darah Tinggi)": {
        "Amlodipine 5mg": {"nama": "Amlodipine 5 mg", "pabrik": "PT Anugrah Amartha Global (OGB Dexa)"},
        "Amlodipine 10mg": {"nama": "Amlodipine 10 mg", "pabrik": "PT Anugrah Amartha Global (OGB Dexa)"},
        "Candesartan Cilexetil 8mg": {"nama": "Candesartan Cilexetil 8 mg", "pabrik": "PT Pharmacon (OGB Dexa)"},
        "Concor 2.5mg": {"nama": "Concor (Bisoprolol Fumarate) 2.5 mg", "pabrik": "PT Merck"},
        "Irbesartan 150mg": {"nama": "Irbesartan 150 mg", "pabrik": "Novell PT Pharmaceutical Lab"}
    },
    "Antiplatelet (Pengencer Darah)": {
        "Ascardia 80mg": {"nama": "Ascardia (Acetylsalicylic Acid) 80 mg", "pabrik": "Pharos Jkt"},
        "Aspilets 100mg": {"nama": "Aspilets 100 mg", "pabrik": "Darya Varia"},
        "Thrombo Aspilets 100mg": {"nama": "Thrombo Aspilets 100 mg", "pabrik": "Darya Varia Laboratoria Tbk"}
    },
    "Kolesterol": {
        "Atorvastatin 20mg": {"nama": "Atorvastatin 20 mg", "pabrik": "PT Otto Pharmaceutical"},
        "Simvastatin 10mg": {"nama": "Simvastatin 10 mg", "pabrik": "PT Dexa Medika (OGB Dexa)"},
        "Simvastatin 20mg": {"nama": "Simvastatin 20 mg", "pabrik": "Hexpharmjaya (HJ)"}
    },
    "Tiroid": {
        "Eutyrox 100mcg": {"nama": "Euthyrox 100 microgram", "pabrik": "Merck Healthcare"}
    },
    "Diabetes": {
        "Forxiga 10mg": {"nama": "Forxiga (Dapagliflozin) 10 mg", "pabrik": "AstraZeneca"},
        "Glimepiride 1mg": {"nama": "Glimepiride 1 mg", "pabrik": "PT Beta Pharmacon (OGB Dexa)"},
        "Metformin 500mg": {"nama": "Metformin 500 mg", "pabrik": "Hexpharmjaya (HJ)"}
    },
    "Asam Urat": {
        "Allopurinol 100mg": {"nama": "Allopurinol 100 mg", "pabrik": "Hexpharmjaya (HJ)"}
    },
    "Alergi": {
        "Loratadine 10mg": {"nama": "Loratadine 10 mg", "pabrik": "Hexpharmjaya (HJ)"}
    },
    "Vitamin / Saraf": {
        "Mecobalamin 500mcg": {"nama": "Mecobalamin 500 mcg", "pabrik": "PT Etercon Pharma Novell"}
    },
    "Sendi & Tulang": {
        "Calcifar Plus": {"nama": "Calcifar Plus (Kalsium + Vitamin D3)", "pabrik": "PT Ifars"},
        "Glucosamine MPL 500mg": {"nama": "Glucosamine MPL 500 mg (Sendi)", "pabrik": "Medikon"}
    }
}

# Flatten dictionary agar mudah dipanggil saat prediksi sistem
OBAT_INFO = {key: val for kategori in KATEGORI_OBAT.values() for key, val in kategori.items()}

# ===== FUNGSI EKSTRAKSI FITUR =====
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


# ===== UI SIDEBAR (PANDUAN & LIST OBAT) =====
with st.sidebar:
    st.header("📋 Panduan Penggunaan (SOP)")
    st.markdown("""
    Agar hasil akurat, mohon perhatikan hal berikut:
    1. Jarak lensa kamera ke objek **25 cm**.
    2. Posisi Kamera **tegak lurus 90°** di atas obat.
    3. Sebaiknya hanya ada **1 objek** dalam frame.
    4. Hindari bayangan masuk ke dalam frame.
    5. Jaga pencahayaan agar tidak terlalu silau (*overexposure*) atau terlalu gelap.
    """)
    
    st.divider()
    
    st.header("💊 Daftar 20 Kelas Obat")
    st.write("Sistem dilatih mengenali obat kronis berikut:")
    
    # Loop menampilkan obat berdasarkan kategori penyakitnya di sidebar
    for kategori, list_obat in KATEGORI_OBAT.items():
        with st.expander(kategori):
            for key, info in list_obat.items():
                st.markdown(f"- **{info['nama']}**<br><sub style='color: gray;'>{info['pabrik']}</sub>", unsafe_allow_html=True)


# ===== UI MAIN (USER INTERFACE) =====
st.title("💊 Sistem Identifikasi Sisa Obat")
st.markdown("Bantu pastikan identitas obat lansia agar tidak terbuang sia-sia.")
st.divider()

warna_pilihan = st.selectbox("1. Warna alas yang dipakai saat memfoto:", list(COLOR_RANGES.keys()))
cfg = COLOR_RANGES[warna_pilihan]
uploaded_file = st.file_uploader("2. Unggah foto obat di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    # Mulai Pemrosesan (Berjalan di latar belakang)
    img_array = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (600, 800))

    mask = segment_object(img_bgr, cfg['lower'], cfg['upper'])
    shape = extract_shape_features(mask)

    st.divider()
    
    # Membuat layout 2 kolom: Kiri untuk Gambar, Kanan untuk Hasil Identifikasi
    col_img, col_res = st.columns([1, 1.2])

    with col_img:
        st.image(pil_img, caption="Foto yang diunggah", use_container_width=True)

    with col_res:
        if shape is None:
            st.error("❌ Sistem tidak dapat menemukan objek obat di dalam foto. Pastikan obat tidak tertutup bayangan gelap dan warna alas sudah benar.")
        else:
            color = extract_color_features(img_bgr, mask)
            texture = extract_texture_features(img_bgr, mask)
            fitur = np.array(list(shape) + list(color) + list(texture)).reshape(1, -1)

            knn, scaler = load_model(cfg['key'])
            fitur_scaled = scaler.transform(fitur)

            distance, _ = knn.kneighbors(fitur_scaled, n_neighbors=1)
            distance = distance[0][0]
            
            # Konversi jarak Euclidean ke Confidence Score %
            conf_score = max(0.0, (1 - (distance / cfg['threshold'])) * 100)

            # PENENTUAN HASIL
            if distance > cfg['threshold']:
                st.error("⚠️ **OBAT TIDAK DIKENALI**")
                st.write("Sistem mendeteksi bahwa objek ini **bukan** salah satu dari 20 obat yang kami kenali, ATAU foto diambil dari jarak/pencahayaan yang tidak sesuai SOP.")
            else:
                pred_subkelas = knn.predict(fitur_scaled)[0]
                
                # Pembersihan nama obat untuk mencocokkan kamus
                pred_obat_raw = re.sub(r'_\d$', '', pred_subkelas).replace('_', ' ')
                obat_data = OBAT_INFO.get(pred_obat_raw, {"nama": pred_obat_raw, "pabrik": "Tidak diketahui"})
                
                # Cek Kondisi Fisik
                fase_match = re.search(r'_(\d)$', pred_subkelas)
                fase_teks = "Kondisi tidak spesifik"
                if fase_match:
                    fase_num = fase_match.group(1)
                    if fase_num == '1': fase_teks = "Kemasan Utuh (Fase 1)"
                    elif fase_num == '2': fase_teks = "Kemasan Lecek/Penyok (Fase 2)"
                    elif fase_num == '3': fase_teks = "Kemasan Terpotong (Fase 3)"
                    elif fase_num == '4': fase_teks = "Pil Telanjang tanpa kemasan (Fase 4)"

                # Tampilan Hasil Sukses di Samping Citra
                st.success("✅ **IDENTIFIKASI BERHASIL**")
                st.markdown(f"## {obat_data['nama']}")
                st.write(f"🏢 **Diproduksi oleh:** {obat_data['pabrik']}")
                st.write(f"🔍 **Kondisi Fisik Saat Difoto:** {fase_teks}")
                st.write(f"🎯 **Tingkat Keyakinan Sistem:** {conf_score:.1f}%")
                
    # Expander Data Teknis berada di bawah gambar (Full Width)
    if shape is not None:
        st.write("")
        with st.expander("🛠️ Data Teknis (Untuk Keperluan Peneliti/Sidang)"):
            st.write("Mask segmentasi dan 11 nilai fitur ekstraksi:")
            st.image(mask, caption="Hasil Mask Biner OpenCV", width=200, clamp=True)
            
            # Dibagi kembali menjadi 3 Kolom untuk Fitur
            col_morf, col_hsv, col_glcm = st.columns(3)
            
            with col_morf:
                st.markdown("**Bentuk (Morfologi)**")
                st.code(f"Area  : {shape[0]:.2f}\nPerim : {shape[1]:.2f}\nCirc  : {shape[2]:.4f}\nRatio : {shape[3]:.4f}")
            
            with col_hsv:
                st.markdown("**Warna (HSV)**")
                st.code(f"Hue : {color[0]:.2f}\nSat : {color[1]:.2f}\nVal : {color[2]:.2f}")
            
            with col_glcm:
                st.markdown("**Tekstur (GLCM)**")
                st.code(f"Cont : {texture[0]:.4f}\nCorr : {texture[1]:.4f}\nEner : {texture[2]:.4f}\nHomo : {texture[3]:.4f}")