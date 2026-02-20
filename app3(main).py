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

# ==========================================
# 1. การตั้งค่าระบบกล้องเครือข่าย (Multi-Camera Config)
# ==========================================
CAMERAS_CONFIG = {
    "Camera 1 (Front)": "http://192.168.0.117/capture",
    "Camera 2 (Back)": "http://192.168.0.118/capture" # แก้ IP ให้ตรงกับกล้องตัวที่ 2
}
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:latest" 
DETECTION_COOLDOWN = 10 # หน่วงเวลาบันทึกประวัติ 10 วินาที เพื่อไม่ให้ Log ล้น

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

# ==========================================
# 2. การจัดการสถานะ (Session State)
# ==========================================
if 'system_running' not in st.session_state:
    st.session_state.system_running = False
if 'is_thinking' not in st.session_state:
    st.session_state.is_thinking = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = [] # ถังเก็บประวัติการตรวจจับ

# โครงสร้างหน่วยความจำแยกตามกล้อง
if 'cameras_data' not in st.session_state:
    st.session_state.cameras_data = {
        name: {"frame": None, "status": "Offline 🔴", "last_detect_time": 0} 
        for name in CAMERAS_CONFIG.keys()
    }

# ==========================================
# 3. Thread 1: กล้อง + YOLO (ทำงานแยกตามกล้อง)
# ==========================================
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
                
                # ระบบบันทึกประวัติ (Debouncing Logic)
                if person_detected_in_frame:
                    current_time = time.time()
                    last_detect = st.session_state.cameras_data[cam_name]["last_detect_time"]
                    
                    # ถ้าเจอคน และเวลาผ่านไปมากกว่า Cooldown ค่อยบันทึกใหม่
                    if current_time - last_detect > DETECTION_COOLDOWN:
                        log_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.detection_log.insert(0, f"[{log_time_str}] 🚨 พบคนวิ่งผ่าน {cam_name}")
                        st.session_state.cameras_data[cam_name]["last_detect_time"] = current_time

                # อัปเดตภาพ
                st.session_state.cameras_data[cam_name]["frame"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                st.session_state.cameras_data[cam_name]["status"] = "Offline 🔴"
                
        except Exception as e:
            st.session_state.cameras_data[cam_name]["status"] = "Offline 🔴"
        
        time.sleep(0.05)

# ==========================================
# 4. Thread 2: คุยกับ Ollama 
# ==========================================
# ==========================================
# 4.1 ระบบ RAG (Vector Database & Embedding)
# ==========================================
def get_embedding(text):
    """ส่งข้อความไปแปลงเป็นพิกัดเวกเตอร์ผ่าน Ollama"""
    response = requests.post("http://localhost:11434/api/embeddings", 
                             json={"model": "nomic-embed-text", "prompt": text})
    return np.array(response.json()["embedding"])

def process_uploaded_file(uploaded_file):
    """อ่านไฟล์และหั่นเป็น Vector Database จำลอง"""
    text = ""
    # 1. Parsing: อ่านข้อความจากไฟล์
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = uploaded_file.getvalue().decode("utf-8")
        
    # 2. Chunking: หั่นข้อความเป็นท่อนๆ (ย่อหน้าละประมาณ 200 ตัวอักษร)
    words = text.split()
    chunks = [' '.join(words[i:i+50]) for i in range(0, len(words), 50)]
    
    # 3. Embedding: แปลงทุกท่อนเป็นตัวเลขแล้วเก็บใส่ State
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
        
    # แปลงคำถามเป็นเวกเตอร์
    query_vector = get_embedding(query)
    
    # คำนวณความเหมือน (Dot Product)
    scores = []
    for item in st.session_state.vector_db:
        # คณิตศาสตร์หาความใกล้เคียงของเวกเตอร์
        similarity = np.dot(query_vector, item["vector"]) / (np.linalg.norm(query_vector) * np.linalg.norm(item["vector"]))
        scores.append((similarity, item["text"]))
        
    # เรียงลำดับและดึงข้อความที่คะแนนสูงสุดมาต่อกัน
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

# ==========================================
# 5. ส่วนแสดงผล UI (Layout & Sidebar)
# ==========================================
st.set_page_config(page_title="TwinEyes Hub", layout="wide")

# แถบด้านข้าง: แสดงสถานะกล้องและประวัติ
with st.sidebar:
    st.title("🎛️ Control Panel")
    if st.button("▶️ เริ่มระบบทั้งหมด" if not st.session_state.system_running else "⏹️ ปิดระบบทั้งหมด", use_container_width=True):
        st.session_state.system_running = not st.session_state.system_running
        if st.session_state.system_running:
            # สั่งรัน Thread ตามจำนวนกล้องที่มี
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
    
    # ลบประวัติเก่าถ้าเกิน 50 รายการ (Sliding Window)
    while len(st.session_state.detection_log) > 50:
        st.session_state.detection_log.pop()
        
    log_container = st.container(height=300)
    with log_container:
        if not st.session_state.detection_log:
            st.write("ยังไม่มีเหตุการณ์ผิดปกติ")
        else:
            for log_entry in st.session_state.detection_log:
                st.write(log_entry)

# พื้นที่หลัก
st.title("มองงง เธอสาวเธอสวยฉันจึงได้มองง")

col1, col2 = st.columns([1.5, 1])

# ฝั่งซ้าย: เลือกกล้องและแสดงวิดีโอ
with col1:
    # Dropdown ให้ผู้ใช้เลือกว่าจะดูกล้องไหน
    selected_cam = st.selectbox("📷 เลือกกล้องที่ต้องการดูและวิเคราะห์:", list(CAMERAS_CONFIG.keys()))
    
    frame_placeholder = st.empty()
    active_frame = st.session_state.cameras_data[selected_cam]["frame"]
    
    if st.session_state.system_running and active_frame is not None:
        frame_placeholder.image(active_frame, channels="RGB", use_container_width=True)

        # --- โค้ดปุ่ม Capture ที่เพิ่มเข้ามาใหม่ ---
        if st.button("📸 บันทึกภาพหลักฐาน (Capture)", use_container_width=True):
            try:
                # 1. สร้างโฟลเดอร์ชื่อ 'captures' ในโฟลเดอร์เดียวกับโปรเจกต์ (ถ้ายังไม่มีให้สร้างใหม่)
                os.makedirs("captures", exist_ok=True)
                
                # 2. ดึงเวลาปัจจุบันมาทำเป็นชื่อไฟล์ (เพื่อไม่ให้ชื่อซ้ำกัน)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captures/evidence_{selected_cam}_{timestamp}.jpg"
                
                # 3. แปลงสีจาก RGB กลับเป็น BGR ตามข้อจำกัดของ OpenCV
                bgr_frame = cv2.cvtColor(active_frame, cv2.COLOR_RGB2BGR)
                
                # 4. สั่งเขียนไฟล์ลงฮาร์ดดิสก์
                cv2.imwrite(filename, bgr_frame)
                
                # แจ้งเตือนว่าเซฟสำเร็จ
                st.success(f"บันทึกภาพสำเร็จ: โฟลเดอร์ {filename}")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการบันทึกภาพ: {e}")
        # -----------------------------------

    elif st.session_state.system_running and active_frame is None:
        frame_placeholder.warning(f"กำลังรอสัญญาณภาพจาก {selected_cam}...")

# ฝั่งขวา: ระบบแชท AI
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
            
            # --- RAG Action ---
            # 1. ค้นหากฎที่เกี่ยวกับ "การรักษาความปลอดภัยและคนแปลกหน้า"
            relevant_context = retrieve_relevant_rules("คนแปลกหน้า ผู้บุกรุก กฎความปลอดภัย")
            
            # 2. นำกฎที่ค้นเจอไปแนบกับ System Prompt
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
            
    # 3. ช่องพิมพ์แชทถามคำถามเฉพาะเจาะจง (พร้อมระบบ RAG)
    user_input = st.chat_input("ถามคำถามเกี่ยวกับกล้องที่เลือก...", disabled=st.session_state.is_thinking)
    if user_input:
        if active_frame is not None:
            st.session_state.is_thinking = True
            
            # บันทึกคำถามผู้ใช้ลงประวัติแชทบนหน้าเว็บ
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # --- กลไก RAG สำหรับช่องแชท ---
            # 1. RETRIEVAL: เอาคำถามของผู้ใช้ไปค้นหาในไฟล์ PDF/TXT
            relevant_context = retrieve_relevant_rules(user_input)
            
            # 2. AUGMENTATION: สร้าง System Rule ใหม่โดยฝังกฎเข้าไป (ถ้าค้นเจอ)
            if relevant_context.strip():
                chat_rule = f"""You are a helpful security assistant. 
                Answer the user's question based ONLY on the provided image and the following official rules:
                
                [RULES]
                {relevant_context}
                [/RULES]
                
                Always answer in Thai language. If the image violates the rules, point it out."""
            else:
                # ถ้าไม่ค้นเจอข้อมูลในไฟล์ หรือยังไม่ได้อัปโหลดไฟล์ ให้ตอบตามปกติ
                chat_rule = "You are a helpful security assistant. Answer based ONLY on the provided image. Answer in Thai."
            
            # 3. GENERATION: ส่งคำถาม + ภาพ + กฎ (System Prompt) ให้ Gemma 3
            t3 = threading.Thread(target=ask_ollama_async, args=(active_frame, user_input, chat_rule), daemon=True)
            add_script_run_ctx(t3)
            t3.start()
            st.rerun()
        else:
            st.warning("ไม่มีสัญญาณภาพให้วิเคราะห์ครับ")

# รักษาวงรอบการเรนเดอร์ UI
if st.session_state.system_running:
    time.sleep(1.0) 
    st.rerun()