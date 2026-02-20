import PyPDF2
import streamlit as st
import cv2
import requests
import numpy as np
import threading
import time
import base64
import os
from datetime import datetime
from ultralytics import YOLO
from streamlit.runtime.scriptrunner import add_script_run_ctx


CAMERAS_CONFIG = {
    "Camera 1 (Front)": "http://192.168.0.117/capture",
    "Camera 2 (Back)": "http://192.168.0.118/capture" 
}
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:latest" 
DETECTION_COOLDOWN = 10 

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()


if 'system_running' not in st.session_state:
    st.session_state.system_running = False
if 'is_thinking' not in st.session_state:
    st.session_state.is_thinking = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = [] 


if 'cameras_data' not in st.session_state:
    st.session_state.cameras_data = {
        name: {"frame": None, "status": "Offline 🔴", "last_detect_time": 0} 
        for name in CAMERAS_CONFIG.keys()
    }


def fetch_camera_worker(cam_name, url):
    """ฟังก์ชันทำงานเบื้องหลังที่สามารถรันคู่ขนานกันหลายตัวได้"""
    session = requests.Session()
    while st.session_state.system_running:
        try:
            response = session.get(url, timeout=3)
            if response.status_code == 200:
                st.session_state.cameras_data[cam_name]["status"] = "Online 🟢"
                
                image_array = np.array(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(image_array, -1)
                
                # --- Adaptive Threshold ---
                current_hour = datetime.now().hour
                is_daytime = 6 <= current_hour < 18
                conf_threshold = 0.60 if is_daytime else 0.30 
                
                cv2.putText(img, f"{cam_name} | Th: {int(conf_threshold*100)}%", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # --- YOLO Detection & Logging ---
                results = yolo_model(img, stream=True, verbose=False)
                person_detected_in_frame = False
                
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0: 
                            conf = float(box.conf[0])
                            if conf >= conf_threshold: 
                                person_detected_in_frame = True
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, f"Person {int(conf*100)}%", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                
                if person_detected_in_frame:
                    current_time = time.time()
                    last_detect = st.session_state.cameras_data[cam_name]["last_detect_time"]
                    
                    
                    if current_time - last_detect > DETECTION_COOLDOWN:
                        log_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.detection_log.insert(0, f"[{log_time_str}] 🚨 พบคนวิ่งผ่าน {cam_name}")
                        st.session_state.cameras_data[cam_name]["last_detect_time"] = current_time

                
                st.session_state.cameras_data[cam_name]["frame"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                st.session_state.cameras_data[cam_name]["status"] = "Offline 🔴"
                
        except Exception as e:
            st.session_state.cameras_data[cam_name]["status"] = "Offline 🔴"
        
        time.sleep(0.05)


def get_embedding(text):
    """ส่งข้อความไปแปลงเป็นพิกัดเวกเตอร์ผ่าน Ollama"""
    response = requests.post("http://localhost:11434/api/embeddings", 
                             json={"model": "nomic-embed-text", "prompt": text})
    return np.array(response.json()["embedding"])

def process_uploaded_file(uploaded_file):
    """อ่านไฟล์และหั่นเป็น Vector Database จำลอง"""
    text = ""
    
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = uploaded_file.getvalue().decode("utf-8")
        
    
    words = text.split()
    chunks = [' '.join(words[i:i+50]) for i in range(0, len(words), 50)]
    
    
    st.session_state.vector_db = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        st.session_state.vector_db.append({"text": chunk, "vector": vector})
        progress_bar.progress((i + 1) / len(chunks))
        
    st.success(f"อัปโหลดและสร้างฐานข้อมูลสำเร็จ! (แบ่งเป็น {len(chunks)} ส่วน)")

def retrieve_relevant_rules(query, top_k=2):
    """ค้นหากฎที่เกี่ยวข้องที่สุดด้วยคณิตศาสตร์ (Cosine Similarity)"""
    if 'vector_db' not in st.session_state or len(st.session_state.vector_db) == 0:
        return ""
        
    
    query_vector = get_embedding(query)
    
    
    scores = []
    for item in st.session_state.vector_db:
        
        similarity = np.dot(query_vector, item["vector"]) / (np.linalg.norm(query_vector) * np.linalg.norm(item["vector"]))
        scores.append((similarity, item["text"]))
        
    
    scores.sort(key=lambda x: x[0], reverse=True)
    best_chunks = [score[1] for score in scores[:top_k]]
    return " ".join(best_chunks)

def ask_ollama_async(image_matrix, user_prompt, system_rule="Always answer in Thai language. Be helpful and precise."):
    try:
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_matrix, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "model": MODEL_NAME,
            "prompt": user_prompt,
            "system": system_rule,
            "images": [img_base64],
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            answer = response.json().get('response', 'ไม่มีคำตอบ')
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {response.status_code}"})
    except Exception as e:
        st.session_state.chat_history.append({"role": "assistant", "content": f"การเชื่อมต่อ AI ล้มเหลว: {e}"})
    finally:
        st.session_state.is_thinking = False


st.set_page_config(page_title="TwinEyes Hub", layout="wide")


with st.sidebar:
    st.title("🎛️ Control Panel")
    if st.button("▶️ เริ่มระบบทั้งหมด" if not st.session_state.system_running else "⏹️ ปิดระบบทั้งหมด", use_container_width=True):
        st.session_state.system_running = not st.session_state.system_running
        if st.session_state.system_running:
            
            for cam_name, cam_url in CAMERAS_CONFIG.items():
                t = threading.Thread(target=fetch_camera_worker, args=(cam_name, cam_url), daemon=True)
                add_script_run_ctx(t)
                t.start()
        st.rerun()
    st.divider()
    st.subheader("📚 อัปโหลดกฎ/คู่มือ (RAG)")
    uploaded_file = st.file_uploader("รองรับไฟล์ .txt และ .pdf", type=["txt", "pdf"])
    if uploaded_file is not None and st.button("ประมวลผลไฟล์เข้าสู่สมอง AI"):
        with st.spinner("กำลังแปลงข้อความเป็นเวกเตอร์..."):
            process_uploaded_file(uploaded_file)    
    st.divider()
    st.subheader("📶 สถานะกล้อง")
    for name in CAMERAS_CONFIG.keys():
        st.write(f"**{name}:** {st.session_state.cameras_data[name]['status']}")
        
    st.divider()
    st.subheader("📋 ประวัติการตรวจจับ (YOLO Log)")
    
    
    while len(st.session_state.detection_log) > 50:
        st.session_state.detection_log.pop()
        
    log_container = st.container(height=300)
    with log_container:
        if not st.session_state.detection_log:
            st.write("ยังไม่มีเหตุการณ์ผิดปกติ")
        else:
            for log_entry in st.session_state.detection_log:
                st.write(log_entry)


st.title("มองงง เธอสาวเธอสวยฉันจึงได้มองง")

col1, col2 = st.columns([1.5, 1])


with col1:
    
    selected_cam = st.selectbox("📷 เลือกกล้องที่ต้องการดูและวิเคราะห์:", list(CAMERAS_CONFIG.keys()))
    
    frame_placeholder = st.empty()
    active_frame = st.session_state.cameras_data[selected_cam]["frame"]
    
    if st.session_state.system_running and active_frame is not None:
        frame_placeholder.image(active_frame, channels="RGB", use_container_width=True)

        
        if st.button("📸 บันทึกภาพหลักฐาน (Capture)", use_container_width=True):
            try:
                
                os.makedirs("captures", exist_ok=True)
                
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captures/evidence_{selected_cam}_{timestamp}.jpg"
                
                
                bgr_frame = cv2.cvtColor(active_frame, cv2.COLOR_RGB2BGR)
                
                
                cv2.imwrite(filename, bgr_frame)
                
                
                st.success(f"บันทึกภาพสำเร็จ: โฟลเดอร์ {filename}")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการบันทึกภาพ: {e}")
        

    elif st.session_state.system_running and active_frame is None:
        frame_placeholder.warning(f"กำลังรอสัญญาณภาพจาก {selected_cam}...")


with col2:
    st.subheader(f"💬 วิเคราะห์ภาพจาก {selected_cam}")
    
    chat_container = st.container(height=400)
    with chat_container:
        while len(st.session_state.chat_history) > 20:
            st.session_state.chat_history.pop(0)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if st.session_state.is_thinking:
            with st.chat_message("assistant"):
                st.markdown("กำลังวิเคราะห์ภาพ... ⏳")
                
    if st.button("🚨 วิเคราะห์ภาพตามกฎที่อัปโหลด", disabled=st.session_state.is_thinking):
        if active_frame is not None:
            st.session_state.is_thinking = True
            st.session_state.chat_history.append({"role": "user", "content": "วิเคราะห์ภาพพร้อมอ้างอิงกฎของพื้นที่"})
            
            
            
            relevant_context = retrieve_relevant_rules("คนแปลกหน้า ผู้บุกรุก กฎความปลอดภัย")
            
            
            rag_system_rule = f"""You are a strict security AI. 
            Analyze the image based STRICTLY on the following retrieved rules:
            
            [RULES]
            {relevant_context}
            [/RULES]
            
            Always answer in Thai. Explain if the situation violates the rules."""
            
            t2 = threading.Thread(target=ask_ollama_async, args=(active_frame, "วิเคราะห์ภาพนี้อิงตามกฎ", rag_system_rule), daemon=True)
            add_script_run_ctx(t2)
            t2.start()
            st.rerun()
            
    
    user_input = st.chat_input("ถามคำถามเกี่ยวกับกล้องที่เลือก...", disabled=st.session_state.is_thinking)
    if user_input:
        if active_frame is not None:
            st.session_state.is_thinking = True
            
            
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            
            
            relevant_context = retrieve_relevant_rules(user_input)
            
            
            if relevant_context.strip():
                chat_rule = f"""You are a helpful security assistant. 
                Answer the user's question based ONLY on the provided image and the following official rules:
                
                [RULES]
                {relevant_context}
                [/RULES]
                
                Always answer in Thai language. If the image violates the rules, point it out."""
            else:
                
                chat_rule = "You are a helpful security assistant. Answer based ONLY on the provided image. Answer in Thai."
            
            
            t3 = threading.Thread(target=ask_ollama_async, args=(active_frame, user_input, chat_rule), daemon=True)
            add_script_run_ctx(t3)
            t3.start()
            st.rerun()
        else:
            st.warning("ไม่มีสัญญาณภาพให้วิเคราะห์ครับ")


if st.session_state.system_running:
    time.sleep(1.0) 
    st.rerun()