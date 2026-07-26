import os

# ========== AUTO-DOWNLOAD MODEL FROM GOOGLE DRIVE ==========
GDRIVE_FILE_ID = "1XKHJubHZ_o3O_9CufXov267GOXw6EwVf"  # <-- PASTE YOUR FILE ID HERE

if not os.path.exists("best.pt"):
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, "best.pt", quiet=False)
    except Exception as e:
        print(f"Model download failed: {e}")
# ========================================

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

st.set_page_config(page_title="Vehicle Number Plate Detector", page_icon="🚗", layout="centered")

# ==================== CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
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
.main-container { max-width: 800px; margin: 0 auto; padding: 20px; }
.main-title {
    font-family: 'Inter', sans-serif; font-size: 2.6rem; font-weight: 800;
    color: #ffffff; text-align: center; margin-top: 30px; margin-bottom: 6px;
    letter-spacing: -1px;
}
.sub-title {
    font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 400;
    color: #8ba4be; text-align: center; margin-bottom: 30px;
}
.glass-card {
    background: rgba(20, 30, 48, 0.65);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.section-header {
    font-size: 1.2rem; font-weight: 700; color: #ffffff;
    margin-bottom: 6px; display: flex; align-items: center; gap: 10px;
}
.section-desc { font-size: 0.9rem; color: #8ba4be; margin-bottom: 16px; }
.status-box {
    background: rgba(74, 144, 226, 0.12);
    border: 1px solid rgba(74, 144, 226, 0.25);
    border-radius: 10px; padding: 12px 16px;
    color: #8fc1ff; font-size: 0.9rem; margin-top: 12px;
}
.result-card {
    background: rgba(20, 30, 48, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(74, 144, 226, 0.2);
    border-radius: 16px; padding: 24px; margin-top: 20px;
    animation: fadeInUp 0.5s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.download-btn {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #fff; font-weight: 700; font-size: 0.9rem;
    padding: 12px 28px; border-radius: 10px; border: none;
    cursor: pointer; text-decoration: none; margin-top: 14px;
    transition: all 0.3s; box-shadow: 0 4px 15px rgba(16,185,129,0.3);
}
.download-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 25px rgba(16,185,129,0.5); }
.processing-text { color: #8fc1ff; font-size: 0.9rem; text-align: center; font-weight: 500; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==================== FUNCTIONS ====================

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
        if x2 > x1 and y2 > y1 and (x2-x1) > 15 and (y2-y1) > 15:
            detections.append(([x1,y1,x2,y2], conf))
    
    detections.sort(key=lambda x: x[1], reverse=True)
    keep = []
    for d in detections:
        dup = False
        c1 = ((d[0][0]+d[0][2])/2, (d[0][1]+d[0][3])/2)
        for k in keep:
            c2 = ((k[0][0]+k[0][2])/2, (k[0][1]+k[0][3])/2)
            if np.hypot(c1[0]-c2[0], c1[1]-c2[1]) < 50:
                dup = True; break
        if not dup: keep.append(d)
    
    annotated = img.copy()
    for (x1,y1,x2,y2), _ in keep:
        margin = 40
        vx1 = max(0,x1-margin); vy1 = max(0,y1-margin*2)
        vx2 = min(w,x2+margin); vy2 = min(h,y2+margin)
        cv2.rectangle(annotated, (vx1,vy1), (vx2,vy2), (0,255,0), 4)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 3)
        cropped = img[y1:y2, x1:x2]
        if cropped.size > 0:
            thumb = cv2.resize(cropped, (320,160))
            th, tw = thumb.shape[:2]
            tx, ty = x2+25, y1
            if tx+tw >= w: tx = x1-tw-25
            if tx < 0: tx = max(0,vx1); ty = vy1-th-15
            if ty < 0: ty = vy2+15
            if ty+th >= h: ty = h-th-5
            if tx >= 0 and ty >= 0 and tx+tw <= w and ty+th <= h:
                annotated[ty:ty+th, tx:tx+tw] = thumb
                cv2.rectangle(annotated, (tx,ty), (tx+tw,ty+th), (255,255,255), 3)
    return annotated

def process_video(video_bytes, model):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total > 300:
        cap.release()
        os.remove(tfile.name)
        return None, "Video too long. Max 10 seconds allowed."
    
    out_raw = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_raw, fourcc, fps, (w, h))
    
    infer_w = 480
    scale = infer_w / w
    infer_h = int(h * scale)
    SKIP = 3
    
    frame_count = 0
    last_keep = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        orig = frame.copy()
        
        if frame_count % SKIP == 1 or not last_keep:
            small = cv2.resize(frame, (infer_w, infer_h))
            results = model(small, verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                x1, y1 = int(x1/scale), int(y1/scale)
                x2, y2 = int(x2/scale), int(y2/scale)
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(w,x2), min(h,y2)
                if x2>x1 and y2>y1 and (x2-x1)>15 and (y2-y1)>15:
                    detections.append(([x1,y1,x2,y2], conf))
            
            detections.sort(key=lambda x: x[1], reverse=True)
            last_keep = []
            for d in detections:
                dup = False
                c1 = ((d[0][0]+d[0][2])/2, (d[0][1]+d[0][3])/2)
                for k in last_keep:
                    c2 = ((k[0][0]+k[0][2])/2, (k[0][1]+k[0][3])/2)
                    if np.hypot(c1[0]-c2[0], c1[1]-c2[1]) < 50:
                        dup = True; break
                if not dup: last_keep.append(d)
        
        for (x1,y1,x2,y2), _ in last_keep:
            margin = 40
            vx1 = max(0,x1-margin); vy1 = max(0,y1-margin*2)
            vx2 = min(w,x2+margin); vy2 = min(h,y2+margin)
            cv2.rectangle(frame, (vx1,vy1), (vx2,vy2), (0,255,0), 4)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
            
            cropped = orig[y1:y2, x1:x2]
            if cropped.size > 0:
                thumb = cv2.resize(cropped, (320,160))
                th, tw = thumb.shape[:2]
                tx, ty = x2+25, y1
                if tx+tw >= w: tx = x1-tw-25
                if tx < 0: tx = max(0,vx1); ty = vy1-th-15
                if ty < 0: ty = vy2+15
                if ty+th >= h: ty = h-th-5
                if tx >= 0 and ty >= 0 and tx+tw <= w and ty+th <= h:
                    frame[ty:ty+th, tx:tx+tw] = thumb
                    cv2.rectangle(frame, (tx,ty), (tx+tw,ty+th), (255,255,255), 3)
        
        writer.write(frame)
    
    cap.release()
    writer.release()
    os.remove(tfile.name)
    
    final_path = tempfile.mktemp(suffix='.mp4')
    cmd = ['ffmpeg','-y','-i',out_raw,'-c:v','libx264','-pix_fmt','yuv420p',
           '-preset','fast','-crf','23','-movflags','+faststart',final_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if os.path.exists(out_raw): os.remove(out_raw)
    
    if not os.path.exists(final_path) or os.path.getsize(final_path) < 1000:
        return None, "Video conversion failed."
    
    return final_path, None

# ==================== UI ====================

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="main-title">🚗 Vehicle Number Plate Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CNN Architecture Used YOLOv8</div>', unsafe_allow_html=True)

# Model card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
if os.path.exists("best.pt") and os.path.getsize("best.pt") > 1_000_000:
    st.success(f"✅ Model ready: **best.pt** ({os.path.getsize('best.pt')/1024/1024:.1f} MB)")
    model_path = "best.pt"
else:
    if os.path.exists("best.pt"):
        st.warning(f"⚠️ File corrupted ({os.path.getsize('best.pt')} bytes). Re-upload.")
    else:
        st.info("📦 Upload your trained model.")
    model_file = st.file_uploader("Drop **best.pt**", type=["pt"])
    if model_file:
        with open("best.pt","wb") as f: f.write(model_file.getbuffer())
        st.success(f"✅ Uploaded ({os.path.getsize('best.pt')/1024/1024:.1f} MB)")
        model_path = "best.pt"
    else:
        st.error("❌ Please upload **best.pt**"); st.stop()
st.markdown('</div>', unsafe_allow_html=True)

# Upload card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📤 Upload Media</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Select type and upload. For video, use max 10 seconds to avoid timeout.</div>', unsafe_allow_html=True)

input_type = st.selectbox("Input Type", ["Image","Video"], label_visibility="collapsed")

if input_type == "Image":
    uploaded_file = st.file_uploader("Image (JPG, PNG, JPEG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
else:
    uploaded_file = st.file_uploader("Video (MP4, AVI, MOV) — max 10 sec", type=["mp4","avi","mov"], label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

detect_clicked = False
if uploaded_file is not None:
    st.markdown('<div class="glass-card" style="padding:12px;">', unsafe_allow_html=True)
    detect_clicked = st.button("🔍 Detect Licence Plates", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PROCESSING ====================
if detect_clicked and uploaded_file:
    try:
        with st.spinner("🚀 Loading YOLOv8..."):
            model = load_model(model_path)
        
        if input_type == "Image":
            with st.spinner("🔍 Detecting..."):
                file_bytes = uploaded_file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result_img = process_image(img_rgb, model)
            
            pil_res = Image.fromarray(result_img)
            buf = io.BytesIO(); pil_res.save(buf, format="PNG")
            img_bytes = buf.getvalue(); img_b64 = base64.b64encode(img_bytes).decode()
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🎯 Detection Result</div>', unsafe_allow_html=True)
            st.image(result_img, use_column_width=True)
            st.markdown(f'<div style="text-align:center;"><a href="data:image/png;base64,{img_b64}" download="detected_plate.png"><button class="download-btn">⬇️ Download Image</button></a></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🎬 Processing Video</div>', unsafe_allow_html=True)
            
            status_text = st.empty()
            status_text.markdown('<div class="status-box">⏳ Starting...</div>', unsafe_allow_html=True)
            
            video_bytes = uploaded_file.read()
            result_path, error_msg = process_video(video_bytes, model)
            
            status_text.empty()
            
            if error_msg:
                st.error(f"❌ {error_msg}")
                st.markdown('</div>', unsafe_allow_html=True)
            elif result_path and os.path.exists(result_path):
                st.markdown('<div class="section-header" style="margin-top:16px;">✅ Result</div>', unsafe_allow_html=True)
                
                with open(result_path, "rb") as f:
                    vid_bytes = f.read()
                st.video(vid_bytes)
                
                vid_b64 = base64.b64encode(vid_bytes).decode()
                st.markdown(f'<div style="text-align:center;"><a href="data:video/mp4;base64,{vid_b64}" download="detected_plate.mp4"><button class="download-btn">⬇️ Download Video</button></a></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                os.remove(result_path)
            else:
                st.error("❌ Unknown video error.")
                st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
