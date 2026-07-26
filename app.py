import os

# ========== AUTO-DOWNLOAD MODEL FROM GOOGLE DRIVE ==========
GDRIVE_FILE_ID = "1XKHJubHZ_o3O_9CufXov267GOXw6EwVf"  # <-- PASTE YOUR FILE ID HERE

if not os.path.exists("best.pt"):
    import gdown
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, "best.pt", quiet=False)
# ===========================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import subprocess
import io
import base64
import torch

# ========== PYTORCH 2.6 FIX ==========
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# =====================================

from ultralytics import YOLO

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Vehicle Number Plate Detector",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS (MATCHING SCREENSHOT STYLE) ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Animated Background: Black → Navy → Light Blue */
.stApp {
    background: linear-gradient(-45deg, #000000, #0a1628, #1e3a5f, #4a90e2, #0a1628, #000000);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    font-family: 'Inter', sans-serif;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main Container */
.main-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

/* Title Section */
.title-section {
    text-align: center;
    margin-bottom: 10px;
    margin-top: 30px;
}
.main-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    margin-bottom: 8px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.sub-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: #8ba4be;
    letter-spacing: 0.5px;
}

/* Glass Cards (like screenshot) */
.glass-card {
    background: rgba(20, 30, 48, 0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}

/* Upload Zone */
.upload-zone {
    background: rgba(10, 15, 25, 0.5);
    border: 2px dashed rgba(74, 144, 226, 0.3);
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: rgba(74, 144, 226, 0.6);
    background: rgba(10, 15, 25, 0.7);
}

/* Section Headers */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-desc {
    font-size: 0.9rem;
    color: #8ba4be;
    margin-bottom: 16px;
}

/* Detect Button */
.detect-btn {
    background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    padding: 14px 40px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
    letter-spacing: 0.5px;
}
.detect-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(74, 144, 226, 0.5);
    background: linear-gradient(135deg, #5a9ff2 0%, #4a90e2 100%);
}

/* Status Box (like screenshot blue info box) */
.status-box {
    background: rgba(74, 144, 226, 0.15);
    border: 1px solid rgba(74, 144, 226, 0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #8fc1ff;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 16px;
}

/* Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #4a90e2, #63b3ed) !important;
    border-radius: 10px;
}

/* Result Card */
.result-card {
    background: rgba(20, 30, 48, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(74, 144, 226, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-top: 24px;
    animation: fadeInUp 0.6s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Download Button */
.download-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 12px 28px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    text-decoration: none;
    margin-top: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
}
.download-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(16, 185, 129, 0.5);
}

/* Spinner text */
.processing-text {
    color: #8fc1ff;
    font-size: 0.95rem;
    text-align: center;
    font-weight: 500;
}

/* Hide Streamlit branding */
#MainMenu, footer, header, .stDeployButton {display: none !important;}
</style>
""", unsafe_allow_html=True)

# ==================== FUNCTIONS (DEFINED BEFORE USE) ====================

@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

def process_image(img, model):
    h, w = img.shape[:2]
    results = model(img, verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1 and (x2 - x1) > 15 and (y2 - y1) > 15:
            detections.append(([x1, y1, x2, y2], conf))
    
    detections.sort(key=lambda x: x[1], reverse=True)
    keep = []
    for d in detections:
        dup = False
        c1 = ((d[0][0] + d[0][2]) / 2, (d[0][1] + d[0][3]) / 2)
        for k in keep:
            c2 = ((k[0][0] + k[0][2]) / 2, (k[0][1] + k[0][3]) / 2)
            if np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < 50:
                dup = True
                break
        if not dup:
            keep.append(d)
    
    annotated = img.copy()
    for (x1, y1, x2, y2), _ in keep:
        margin = 40
        vx1 = max(0, x1 - margin)
        vy1 = max(0, y1 - margin * 2)
        vx2 = min(w, x2 + margin)
        vy2 = min(h, y2 + margin)
        
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 255, 0), 4)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        cropped = img[y1:y2, x1:x2]
        if cropped.size > 0:
            thumb = cv2.resize(cropped, (320, 160))
            th, tw = thumb.shape[:2]
            tx, ty = x2 + 25, y1
            
            if tx + tw >= w:
                tx = x1 - tw - 25
            if tx < 0:
                tx = max(0, vx1)
                ty = vy1 - th - 15
            if ty < 0:
                ty = vy2 + 15
            if ty + th >= h:
                ty = h - th - 5
            
            if tx >= 0 and ty >= 0 and tx + tw <= w and ty + th <= h:
                annotated[ty:ty+th, tx:tx+tw] = thumb
                cv2.rectangle(annotated, (tx, ty), (tx+tw, ty+th), (255, 255, 255), 3)
    
    return annotated

def process_video(uploaded_file, model, progress_bar, status_text):
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # For faster inference, scale down to max 640 width
    scale = min(1.0, 640 / w)
    infer_w = int(w * scale)
    infer_h = int(h * scale)
    
    out_raw = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_raw, fourcc, fps, (w, h))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        orig = frame.copy()
        
        # Resize for faster inference
        small = cv2.resize(frame, (infer_w, infer_h))
        results = model(small, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            # Scale back to original resolution
            x1, y1 = int(x1 / scale), int(y1 / scale)
            x2, y2 = int(x2 / scale), int(y2 / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1 and (x2 - x1) > 15 and (y2 - y1) > 15:
                detections.append(([x1, y1, x2, y2], conf))
        
        detections.sort(key=lambda x: x[1], reverse=True)
        keep = []
        for d in detections:
            dup = False
            c1 = ((d[0][0] + d[0][2]) / 2, (d[0][1] + d[0][3]) / 2)
            for k in keep:
                c2 = ((k[0][0] + k[0][2]) / 2, (k[0][1] + k[0][3]) / 2)
                if np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < 50:
                    dup = True
                    break
            if not dup:
                keep.append(d)
        
        for (x1, y1, x2, y2), _ in keep:
            margin = 40
            vx1 = max(0, x1 - margin)
            vy1 = max(0, y1 - margin * 2)
            vx2 = min(w, x2 + margin)
            vy2 = min(h, y2 + margin)
            
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            cropped = orig[y1:y2, x1:x2]
            if cropped.size > 0:
                thumb = cv2.resize(cropped, (320, 160))
                th, tw = thumb.shape[:2]
                tx, ty = x2 + 25, y1
                
                if tx + tw >= w:
                    tx = x1 - tw - 25
                if tx < 0:
                    tx = max(0, vx1)
                    ty = vy1 - th - 15
                if ty < 0:
                    ty = vy2 + 15
                if ty + th >= h:
                    ty = h - th - 5
                
                if tx >= 0 and ty >= 0 and tx + tw <= w and ty + th <= h:
                    frame[ty:ty+th, tx:tx+tw] = thumb
                    cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (255, 255, 255), 3)
        
        writer.write(frame)
        
        # Update progress every frame
        progress = min(frame_count / total, 1.0) if total > 0 else 0
        progress_bar.progress(progress, text=f"Processing frame {frame_count}/{total}")
        
        if frame_count % 30 == 0:
            status_text.markdown(f'<div class="processing-text">⏳ Processed {frame_count} of {total} frames...</div>', unsafe_allow_html=True)
    
    cap.release()
    writer.release()
    os.remove(tfile.name)
    
    status_text.markdown('<div class="processing-text">🎬 Converting to MP4 format...</div>', unsafe_allow_html=True)
    
    final_path = tempfile.mktemp(suffix='.mp4')
    cmd = [
        'ffmpeg', '-y', '-i', out_raw,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'fast', '-crf', '23',
        '-movflags', '+faststart',
        final_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if os.path.exists(out_raw):
        os.remove(out_raw)
    
    return final_path

# ==================== UI STARTS HERE ====================

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title Section
st.markdown('''
    <div class="title-section">
        <div class="main-title">🚗 Vehicle Number Plate Detector</div>
        <div class="sub-title">CNN Architecture Used YOLOv8</div>
    </div>
''', unsafe_allow_html=True)

# Model Status Card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
if os.path.exists("best.pt") and os.path.getsize("best.pt") > 1_000_000:
    st.success(f"✅ Model loaded: **best.pt** ({os.path.getsize('best.pt')/1024/1024:.1f} MB)")
    model_path = "best.pt"
else:
    if os.path.exists("best.pt"):
        st.warning(f"⚠️ File too small ({os.path.getsize('best.pt')} bytes) — likely corrupted.")
    else:
        st.info("📦 Please upload your trained model file.")
    
    model_file = st.file_uploader("Drop **best.pt** here", type=["pt"])
    if model_file:
        with open("best.pt", "wb") as f:
            f.write(model_file.getbuffer())
        st.success(f"✅ Model uploaded! ({os.path.getsize('best.pt')/1024/1024:.1f} MB)")
        model_path = "best.pt"
    else:
        st.error("❌ Please upload **best.pt** to continue")
        st.stop()
st.markdown('</div>', unsafe_allow_html=True)

# Upload Card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📤 Upload Media</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Select input type and upload your file for detection.</div>', unsafe_allow_html=True)

input_type = st.selectbox("Input Type", ["Image", "Video"], label_visibility="collapsed")

if input_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
else:
    uploaded_file = st.file_uploader(
        "Upload Video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Detect Button
detect_clicked = False
if uploaded_file is not None:
    st.markdown('<div class="glass-card" style="padding-top:10px; padding-bottom:10px;">', unsafe_allow_html=True)
    detect_clicked = st.button("🔍 Detect Licence Plates", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PROCESSING ====================
if detect_clicked and uploaded_file:
    with st.spinner("Initializing YOLOv8..."):
        model = load_model(model_path)
    
    if input_type == "Image":
        with st.spinner("Analyzing image..."):
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result_img = process_image(img_rgb, model)
        
        pil_res = Image.fromarray(result_img)
        buf = io.BytesIO()
        pil_res.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎯 Detection Result</div>', unsafe_allow_html=True)
        st.image(result_img, use_column_width=True)
        
        st.markdown(f'''
            <div style="text-align:center;">
                <a href="data:image/png;base64,{img_b64}" download="detected_plate.png">
                    <button class="download-btn">⬇️ Download Processed Image</button>
                </a>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # VIDEO
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎬 Processing Video</div>', unsafe_allow_html=True)
        
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()
        
        result_path = process_video(uploaded_file, model, progress_bar, status_text)
        
        progress_bar.empty()
        status_text.empty()
        
        if result_path and os.path.exists(result_path):
            with open(result_path, "rb") as f:
                video_bytes = f.read()
            
            st.markdown('<div class="section-header" style="margin-top:20px;">✅ Detection Result</div>', unsafe_allow_html=True)
            st.video(video_bytes)
            
            vid_b64 = base64.b64encode(video_bytes).decode()
            st.markdown(f'''
                <div style="text-align:center;">
                    <a href="data:video/mp4;base64,{vid_b64}" download="detected_plate.mp4">
                        <button class="download-btn">⬇️ Download Processed Video</button>
                    </a>
                </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            os.remove(result_path)
        else:
            st.error("❌ Video processing failed. Please try again.")
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
