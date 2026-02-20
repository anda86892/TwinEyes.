import streamlit as st
import requests
import ollama
from PIL import Image
from io import BytesIO
import os

# --- 1. การตั้งค่าโมเดล ---
# ในเมื่อคุณ pull ได้แล้ว ใช้ชื่อนี้ได้เลยครับ!
MODEL_NAME = "gemma3" 

# IP ของกล้อง T-SIMCAM
CAM_A_URL = "http://192.168.1.100/capture" 
CAM_B_URL = "http://192.168.1.101/capture"

# --- 2. ฟังก์ชันเสริม ---
def get_dorm_rules():
    """ดึงกฎจาก rules.txt"""
    if os.path.exists("rules.txt"):
        try:
            with open("rules.txt", "r", encoding="utf-8") as f:
                return f.read()
        except: pass
    return "ไม่มีกฎระเบียบระบุไว้"

def fetch_image(url):
    """ดึงภาพจากกล้อง ESP32"""
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except: return None

# --- 3. หน้าจอ Streamlit ---
st.set_page_config(page_title="TwinEyes: Gemma 3 Guardian", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

st.title("👁️ TwinEyes: Smart Monitoring (Gemma 3 Local)")
st.info(f"กำลังใช้งานโมเดล: {MODEL_NAME} (Native Multimodal)")

col_dash, col_chat = st.columns([0.6, 0.4])

# --- [ฝั่งซ้าย] MONITORING ---
with col_dash:
    st.header("📸 Live Monitoring")
    img_a = fetch_image(CAM_A_URL)
    
    if img_a:
        # บีบขนาดภาพเล็กน้อยเพื่อความเร็วในการส่งให้ Ollama
        img_a.thumbnail((800, 600)) 
        st.image(img_a, use_container_width=True, caption="ภาพสดจากกล้อง A")
        st.session_state.last_frame = img_a
    else:
        st.error("กล้อง Offline หรือ IP ไม่ถูกต้อง")

# --- [ฝั่งขวา] CHAT WITH GEMMA 3 ---
with col_chat:
    st.header("🤖 AI Guardian")
    
    chat_container = st.container(height=500)
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if prompt := st.chat_input("พิมพ์คำถามที่นี่..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Gemma 3 กำลังประมวลผลภาพ..."):
                    rules = get_dorm_rules()
                    
                    # สร้าง System Prompt ให้ Gemma 3 เข้าใจบทบาท
                    context_prompt = f"""
                    คุณคือ AI ผู้ดูแลหอพักที่ชาญฉลาด
                    นี่คือกฎระเบียบที่คุณต้องใช้ตัดสิน: {rules}
                    
                    ภารกิจ: วิเคราะห์ภาพที่ได้รับ และตอบคำถามผู้ใช้เป็นภาษาไทย
                    คำถาม: {prompt}
                    """
                    
                    try:
                        if st.session_state.last_frame:
                            # แปลงภาพเป็น Bytes เพื่อส่งให้ Ollama
                            buf = BytesIO()
                            st.session_state.last_frame.save(buf, format='JPEG')
                            img_bytes = buf.getvalue()
                            
                            response = ollama.chat(
                                model=MODEL_NAME,
                                messages=[{'role': 'user', 'content': context_prompt, 'images': [img_bytes]}]
                            )
                        else:
                            response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': context_prompt}])
                        
                        res_text = response['message']['content']
                    except Exception as e:
                        res_text = f"เกิดข้อผิดพลาด: {str(e)}"
                
                st.markdown(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text})