import os

# ========== AUTO-DOWNLOAD MODEL FROM GOOGLE DRIVE ==========
GDRIVE_FILE_ID = "1XKHJubHZ_o3O_9CufXov267GOXw6EwVf"  # <-- PASTE YOUR FILE ID HERE

if not os.path.exists("best.pt"):
    import gdown
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, "best.pt", quiet=False)
    print(f"✅ Model downloaded: {os.path.getsize('best.pt')} bytes")
else:
    print(f"✅ Model exists: {os.path.getsize('best.pt')} bytes")
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

# Monkey patch for PyTorch 2.6
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO
# ... rest of your code
# ... rest of your code stays exactly the same
# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Vehicle Number Plate Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS (3D + ANIMATIONS) ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #050505, #0f0c29, #1a1a2e, #16213e, #0f3460, #1a1a2e, #050505);
    background-size: 700% 700%;
    animation: bgShift 25s ease infinite;
}

@keyframes bgShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 3D Title */
.title-3d {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.8rem;
    font-weight: 900;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 6px;
    background: linear-gradient(180deg, #00f2fe 0%, #4facfe 50%, #00c6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 10px 20px rgba(0,0,0,0.5));
    text-shadow: 
        0 1px 0 #ccc, 0 2px 0 #c9c9c9, 0 3px 0 #bbb,
        0 4px 0 #b9b9b9, 0 5px 0 #aaa, 0 6px 1px rgba(0,0,0,.1),
        0 0 5px rgba(0,0,0,.1), 0 1px 3px rgba(0,0,0,.3),
        0 3px 5px rgba(0,0,0,.2), 0 5px 10px rgba(0,0,0,.25),
        0 10px 10px rgba(0,0,0,.2), 0 20px 20px rgba(0,0,0,.15),
        0 0 30px rgba(0, 242, 254, 0.4), 0 0 60px rgba(79, 172, 254, 0.2);
    animation: titleFloat 5s ease-in-out infinite, titleGlow 3s ease-in-out infinite alternate;
    transform-style: preserve-3d;
    perspective: 1000px;
    margin-top: 20px;
    margin-bottom: 5px;
}

@keyframes titleFloat {
    0%, 100% { transform: translateY(0px) rotateX(0deg) rotateY(0deg); }
    25% { transform: translateY(-10px) rotateX(3deg) rotateY(-3deg); }
    50% { transform: translateY(-18px) rotateX(0deg) rotateY(0deg); }
    75% { transform: translateY(-10px) rotateX(-3deg) rotateY(3deg); }
}

@keyframes titleGlow {
    0% { filter: brightness(1) drop-shadow(0 0 20px rgba(0,242,254,0.3)); }
    100% { filter: brightness(1.4) drop-shadow(0 0 50px rgba(0,242,254,0.7)); }
}

/* Subtitle */
.subtitle-3d {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    text-align: center;
    color: #8899ac;
    letter-spacing: 10px;
    text-transform: uppercase;
    animation: subtitlePulse 3s ease-in-out infinite;
    margin-bottom: 50px;
}

@keyframes subtitlePulse {
    0%, 100% { opacity: 0.5; letter-spacing: 10px; text-shadow: 0 0 10px rgba(79,172,254,0); }
    50% { opacity: 1; letter-spacing: 14px; color: #4facfe; text-shadow: 0 0 20px rgba(79,172,254,0.5); }
}

/* Glass Container */
.glass-box {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 30px;
    padding: 40px 50px;
    box-shadow: 
        0 8px 40px 0 rgba(0, 0, 0, 0.5),
        inset 0 0 30px rgba(255, 255, 255, 0.03);
    animation: boxFloat 8s ease-in-out infinite;
    max-width: 900px;
    margin: 0 auto;
}

@keyframes boxFloat {
    0%, 100% { transform: translateY(0px); box-shadow: 0 8px 40px rgba(0,0,0,0.5); }
    50% { transform: translateY(-8px); box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 40px rgba(0,242,254,0.05); }
}

/* Animated Border Box */
.animated-border {
    position: relative;
    border-radius: 20px;
    padding: 30px;
    background: rgba(0, 0, 0, 0.2);
    overflow: hidden;
}
.animated-border::before {
    content: '';
    position: absolute;
    top: -2px; left: -2px; right: -2px; bottom: -2px;
    background: linear-gradient(45deg, #00f2fe, #4facfe, #00f2fe, #4facfe);
    background-size: 300% 300%;
    z-index: -1;
    border-radius: 22px;
    animation: borderGlow 4s linear infinite;
}
@keyframes borderGlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Detect Button */
.detect-btn-container {
    text-align: center;
    margin-top: 30px;
}
.detect-btn-container button {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
    color: #000 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 900 !important;
    padding: 16px 50px !important;
    border: none !important;
    border-radius: 50px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.4), inset 0 0 10px rgba(255,255,255,0.3) !important;
    transition: all 0.3s ease !important;
    animation: btnPulse 2.5s infinite;
}
.detect-btn-container button:hover {
    transform: translateY(-3px) scale(1.08) !important;
    box-shadow: 0 0 50px rgba(0, 242, 254, 0.8), inset 0 0 20px rgba(255,255,255,0.5) !important;
}
@keyframes btnPulse {
    0%, 100% { box-shadow: 0 0 25px rgba(0,242,254,0.4); }
    50% { box-shadow: 0 0 50px rgba(0,242,254,0.8); }
}

/* Result Box */
.result-glass {
    background: rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 242, 254, 0.25);
    border-radius: 24px;
    padding: 35px;
    margin-top: 40px;
    animation: resultSlide 1s ease-out, resultGlow 4s ease-in-out infinite alternate;
    box-shadow: 0 0 40px rgba(0, 242, 254, 0.08);
}

@keyframes resultSlide {
    from { opacity: 0; transform: translateY(40px) scale(0.95); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes resultGlow {
    from { box-shadow: 0 0 40px rgba(0,242,254,0.05); border-color: rgba(0,242,254,0.2); }
    to { box-shadow: 0 0 60px rgba(0,242,254,0.15); border-color: rgba(0,242,254,0.4); }
}

/* Download Button */
.download-btn {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: #000;
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    padding: 14px 35px;
    border-radius: 30px;
    border: none;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.95rem;
    box-shadow: 0 0 20px rgba(56, 239, 125, 0.4);
    transition: all 0.3s;
    margin-top: 20px;
    display: inline-block;
    text-decoration: none;
}
.download-btn:hover {
    transform: scale(1.08);
    box-shadow: 0 0 40px rgba(56, 239, 125, 0.8);
}

/* Spinner text animation */
.processing-text {
    font-family: 'Orbitron', sans-serif;
    color: #00f2fe;
    font-size: 1.1rem;
    text-align: center;
    animation: textFlicker 1.5s infinite;
    text-shadow: 0 0 10px rgba(0,242,254,0.5);
}
@keyframes textFlicker {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Sidebar */
.css-1d391kg, section[data-testid="stSidebar"] {
    background: rgba(5, 5, 5, 0.85) !important;
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(0,242,254,0.1);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR: MODEL UPLOAD ====================
with st.sidebar:
    st.markdown('<h2 style="font-family:Orbitron; color:#00f2fe; text-align:center;">⚙️ CONFIG</h2>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(0,242,254,0.2);">', unsafe_allow_html=True)
    
    if os.path.exists("best.pt"):
        st.success("✅ Model ready: **best.pt**")
        model_path = "best.pt"
    else:
        st.warning("📦 Upload your trained model")
        model_file = st.file_uploader("Drop **best.pt** here", type=["pt"])
        if model_file:
            with open("best.pt", "wb") as f:
                f.write(model_file.getbuffer())
            st.success("✅ Model uploaded!")
            model_path = "best.pt"
        else:
            st.error("❌ Please upload **best.pt** to continue")
            st.stop()

@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

# ==================== MAIN UI ====================
st.markdown('<div class="title-3d">Vehicles Number Plate Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-3d">CNN Architecture Used YOLOv8</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# Input type dropdown
input_type = st.selectbox(
    "📂 SELECT INPUT TYPE",
    ["Image", "Video"],
    help="Choose whether to upload an image or a video file"
)

# File uploader based on type
if input_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, PNG, JPEG"
    )
else:
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        help="Supported formats: MP4, AVI, MOV"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ==================== DETECT BUTTON ====================
detect_clicked = False
if uploaded_file is not None:
    st.markdown('<div class="detect-btn-container">', unsafe_allow_html=True)
    detect_clicked = st.button("🔍 Detect Licence Plates", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PROCESSING ====================
if detect_clicked and uploaded_file:
    # Load model once
    with st.spinner("🚀 Initializing YOLOv8 Neural Network..."):
        model = load_model(model_path)
    
    if input_type == "Image":
        with st.spinner("🔍 Analyzing image..."):
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result_img = process_image(img_rgb, model)
        
        # Encode for display & download
        pil_res = Image.fromarray(result_img)
        buf = io.BytesIO()
        pil_res.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        st.markdown('<div class="result-glass">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-family:Orbitron; color:#00f2fe; text-align:center; margin-bottom:20px;">🎯 DETECTION RESULT</h3>', unsafe_allow_html=True)
        st.image(result_img, use_column_width=True)
        
        # Download
        st.markdown(f'''
            <div style="text-align:center;">
                <a href="data:image/png;base64,{img_b64}" download="detected_plate.png">
                    <button class="download-btn">⬇️ Download Processed Image</button>
                </a>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # VIDEO
        status_text = st.empty()
        
        with st.spinner(""):
            status_text.markdown('<div class="processing-text">⏳ Processing video... Please wait.</div>', unsafe_allow_html=True)
            result_path = process_video(uploaded_file, model, status_text)
            status_text.empty()
        
        if result_path and os.path.exists(result_path):
            with open(result_path, "rb") as f:
                video_bytes = f.read()
            
            st.markdown('<div class="result-glass">', unsafe_allow_html=True)
            st.markdown('<h3 style="font-family:Orbitron; color:#00f2fe; text-align:center; margin-bottom:20px;">🎬 DETECTION RESULT</h3>', unsafe_allow_html=True)
            st.video(video_bytes)
            
            # Download
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

# ==================== PROCESSING FUNCTIONS ====================
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
    
    # Deduplicate by center distance
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

def process_video(uploaded_file, model, status_text):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
        
        results = model(frame, verbose=False)[0]
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
        
        if frame_count % 5 == 0 or frame_count == total:
            status_text.markdown(
                f'<div class="processing-text">⏳ Processing {frame_count}/{total} frames... Please wait.</div>',
                unsafe_allow_html=True
            )
    
    cap.release()
    writer.release()
    os.remove(tfile.name)
    
    status_text.markdown(
        f'<div class="processing-text">✅ Processed {frame_count} frames. Converting to browser-compatible MP4...</div>',
        unsafe_allow_html=True
    )
    
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
