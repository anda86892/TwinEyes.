import streamlit as st
import cv2
import requests
import numpy as np
import threading
import time
import base64
from datetime import datetime
from ultralytics import YOLO
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==========================================
# 1. การตั้งค่าพื้นฐาน
# ==========================================
CAMERA_URL = "http://192.168.0.117/capture" # เปลี่ยนให้ตรงกับบอร์ด ESP32
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:latest" 

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

# ==========================================
# 2. การจัดการสถานะ (Session State)
# ==========================================
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'is_thinking' not in st.session_state:
    st.session_state.is_thinking = False
# เพิ่มหน่วยความจำสำหรับเก็บประวัติแชท
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 

# ==========================================
# 3. Thread 1: กล้อง + YOLO (Adaptive Threshold)
# ==========================================
def fetch_camera_with_yolo():
    session = requests.Session()
    while st.session_state.camera_running:
        try:
            response = session.get(CAMERA_URL, timeout=5)
            if response.status_code == 200:
                image_array = np.array(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(image_array, -1)
                
                # --- กลไก Adaptive Threshold ---
                current_hour = datetime.now().hour
                is_daytime = 6 <= current_hour < 18
                # กลางวัน 60%, กลางคืน 30%
                conf_threshold = 0.60 if is_daytime else 0.30 
                
                # พิมพ์โหมดปัจจุบันโชว์บนหน้าจอ
                mode_text = f"DAY MODE (Th:{int(conf_threshold*100)}%)" if is_daytime else f"NIGHT MODE (Th:{int(conf_threshold*100)}%)"
                cv2.putText(img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                results = yolo_model(img, stream=True, verbose=False)
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0: 
                            conf = float(box.conf[0])
                            # ตรวจสอบว่าผ่านเกณฑ์ตามเวลาหรือไม่
                            if conf >= conf_threshold: 
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, f"Person {int(conf*100)}%", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.session_state.latest_frame = img_rgb
        except Exception as e:
            pass
        time.sleep(0.05) 

# ==========================================
# 4. Thread 2: คุยกับ Ollama (รองรับคำถามจากผู้ใช้)
# ==========================================
def ask_ollama_async(image_matrix, user_prompt="Analyze this image and describe what you see in thai language. Focus on people or abnormal events."):
    try:
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_matrix, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "model": MODEL_NAME,
            "prompt": user_prompt, # ใช้คำถามที่ผู้ใช้พิมพ์มา
            "images": [img_base64],
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            answer = response.json().get('response', 'อ่านคำตอบไม่ได้')
            # นำคำตอบของ AI ไปต่อท้ายในประวัติแชท
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {response.status_code}"})
            
    except Exception as e:
        st.session_state.chat_history.append({"role": "assistant", "content": f"ล้มเหลว: {e}"})
        
    finally:
        st.session_state.is_thinking = False

# ==========================================
# 5. ส่วนแสดงผล UI และแชท (Layout)
# ==========================================
st.set_page_config(page_title="TwinEyes Dashboard", layout="wide")
st.title("👁️ TwinEyes: Vision-Language Assistant")

col1, col2 = st.columns([1.5, 1])

# ฝั่งซ้าย: กล้องวิดีโอ
with col1:
    frame_placeholder = st.empty() 
    if st.session_state.camera_running and st.session_state.latest_frame is not None:
        frame_placeholder.image(st.session_state.latest_frame, channels="RGB", use_container_width=True)
    
    if st.button("▶️ เปิดกล้อง" if not st.session_state.camera_running else "⏹️ ปิดกล้อง"):
        st.session_state.camera_running = not st.session_state.camera_running
        if st.session_state.camera_running:
            t1 = threading.Thread(target=fetch_camera_with_yolo, daemon=True)
            add_script_run_ctx(t1) 
            t1.start()
            st.rerun() 

# ฝั่งขวา: ระบบแชท AI
with col2:
    st.subheader("💬 สนทนากับ Gemma 3")
    
    # 1. กล่องแสดงประวัติการแชท
    chat_container = st.container(height=400)
    with chat_container:

        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history.pop(0)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if st.session_state.is_thinking:
            with st.chat_message("assistant"):
                st.markdown("กำลังวิเคราะห์ภาพ... ⏳")
                
    # 2. ปุ่มวิเคราะห์ด่วน (แบบเก่า)
    if st.button("🚨 สั่งวิเคราะห์ภาพรวมด่วน", disabled=st.session_state.is_thinking):
        if st.session_state.latest_frame is not None:
            st.session_state.is_thinking = True
            default_prompt = "Analyze this image and describe what you see in thai language. Focus on people or abnormal events. reply in thai language."
            st.session_state.chat_history.append({"role": "user", "content": "วิเคราะห์ภาพรวมทั้งหมดให้หน่อย"})
            
            t2 = threading.Thread(target=ask_ollama_async, args=(st.session_state.latest_frame, default_prompt), daemon=True)
            add_script_run_ctx(t2)
            t2.start()
            st.rerun()
            
    # 3. ช่องพิมพ์แชทถามคำถามเฉพาะเจาะจง
    user_input = st.chat_input("ถามคำถามเกี่ยวกับภาพวิดีโอ...", disabled=st.session_state.is_thinking)
    if user_input:
        if st.session_state.latest_frame is not None:
            st.session_state.is_thinking = True
            # บันทึกคำถามผู้ใช้ลงประวัติ
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # ส่งคำถามและภาพให้ AI
            t3 = threading.Thread(target=ask_ollama_async, args=(st.session_state.latest_frame, user_input), daemon=True)
            add_script_run_ctx(t3)
            t3.start()
            st.rerun()
        else:
            st.warning("โปรดเปิดกล้องและรอให้ภาพปรากฏก่อนเริ่มสนทนาครับ")

# ลูปรักษาความลื่นไหลของวิดีโอ (หน่วงเวลาให้ช้าลงนิดนึงเพื่อไม่ให้รบกวนการพิมพ์แชท)
if st.session_state.camera_running and st.session_state.latest_frame is not None:
    time.sleep(0.1)
    st.rerun()