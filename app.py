import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="AI-ColoScan Simple", layout="wide")

# Хранилище для топ-5 обрезанных фото
if 'top_crops' not in st.session_state:
    st.session_state.top_crops = []

@st.cache_resource
def load_model():
    return YOLO('kvasir+polypDB.pt')

model = load_model()

# CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .img-label { 
        text-align: center; font-size: 16px; font-weight: 700; 
        color: #3b82f6; margin-bottom: 5px; 
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 30px;
        font-weight: 900;
        margin: 10px 0;
    }
    .found { background-color: #7f1d1d; color: #f87171; border: 2px solid #ef4444; }
    .not-found { background-color: #064e3b; color: #34d399; border: 2px solid #10b981; }
    .crop-card { border: 1px solid #3b82f6; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 AI-ColoScan: Diagnostic System")
st.divider()

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown('<div class="img-label">INPUT FEED</div>', unsafe_allow_html=True)
        raw_placeholder = st.empty()
    with col_v2:
        st.markdown('<div class="img-label">AI ANALYSIS</div>', unsafe_allow_html=True)
        proc_placeholder = st.empty()

    status_placeholder = st.empty()
    stop_btn = st.button("STOP", use_container_width=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break
        
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Отображение видео
        f_raw = cv2.cvtColor(cv2.resize(frame, (480, 320)), cv2.COLOR_BGR2RGB)
        raw_placeholder.image(f_raw)
        
        f_proc = results[0].plot()
        f_proc = cv2.cvtColor(cv2.resize(f_proc, (480, 320)), cv2.COLOR_BGR2RGB)
        proc_placeholder.image(f_proc)
        
        # Логика статуса и кропов
        if len(results[0].boxes) > 0:
            status_placeholder.markdown('<div class="status-box found">POLYP DETECTED</div>', unsafe_allow_html=True)
            
            # Берем самый уверенный бокс кадра для кропа
            box = results[0].boxes[0]
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].cpu().numpy().astype(int) # координаты [x1, y1, x2, y2]
            
            # Делаем обрезку (crop)
            crop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                # Обновляем ТОП-5 (по уровню уверенности)
                if len(st.session_state.top_crops) < 5 or conf > min(st.session_state.top_crops, key=lambda x: x[0])[0]:
                    st.session_state.top_crops.append((conf, crop_rgb))
                    st.session_state.top_crops = sorted(st.session_state.top_crops, key=lambda x: x[0], reverse=True)[:5]
        else:
            status_placeholder.markdown('<div class="status-box not-found">NO POLYPS</div>', unsafe_allow_html=True)

    cap.release()

    # --- КАРУСЕЛЬ (ТОП-5 КРОПОВ) ---
    st.divider()
    st.subheader("🏆 Top 5 Distinct Detections (Crops)")
    if st.session_state.top_crops:
        cols = st.columns(5)
        for i, (score, crop_img) in enumerate(st.session_state.top_crops):
            with cols[i]:
                st.image(crop_img, use_container_width=True)
                st.caption(f"Detection {i+1}")
    else:
        st.write("Waiting for detections...")
else:
    st.info("Please upload a video to start.")
