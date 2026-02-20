# 👁️ TwinEyes: Edge-to-Local Hybrid Vision-Language System

TwinEyes คือสถาปัตยกรรมระบบรักษาความปลอดภัยอัจฉริยะแบบบูรณาการ ที่ผสานความเร็วในการตรวจจับด้วย Computer Vision (YOLOv8) เข้ากับความสามารถในการวิเคราะห์เชิงลึกและตัดสินใจของ Large Language Model (Gemma 3) พร้อมรองรับกล้องหลายตัว (Multi-Camera Network) และระบบถาม-ตอบอิงคู่มือความปลอดภัย (RAG)

## 🌟 ฟีเจอร์หลัก (Key Features)
- **Multi-Camera Asynchronous Backend:** ดึงภาพและตรวจจับวัตถุ (YOLO) แบบ Real-time ผ่าน Background Threads อิสระ โดยไม่กระทบความเสถียรของหน้า UI
- **Adaptive Thresholding:** ปรับความไวแสงและเกณฑ์การตรวจจับ (Confidence Threshold) อัตโนมัติตามช่วงเวลากลางวัน-กลางคืน
- **Event Debouncing & Logging:** ระบบจัดการ Log อัจฉริยะ ป้องกันข้อมูลขยะล้นระบบ (Spam Prevention) เมื่อตรวจพบเหตุการณ์ซ้ำซ้อน
- **RAG (Retrieval-Augmented Generation):** อัปโหลดคู่มือรักษาความปลอดภัย (PDF/TXT) ให้ AI วิเคราะห์ภาพอิงตามกฎของพื้นที่นั้นๆ (Context-Aware Analysis)
- **Interactive Vision-Language Chat:** สนทนาโต้ตอบกับระบบ AI เพื่อสอบถามรายละเอียดของภาพวิดีโอล่าสุดได้แบบอิสระ

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)
ระบบถูกออกแบบโดยใช้หลักการ **Asynchronous Decoupling** เพื่อแก้ปัญหาคอขวดของการเรนเดอร์ UI:
1. **Edge Node:** ESP32-CAM ทำหน้าที่สตรีมภาพผ่าน HTTP Protocol
2. **Vision Engine:** YOLOv8 รันบน Daemon Thread แยกอิสระ เพื่อกวาดสายตาจับผิดปกติที่ 10-20 FPS
3. **AI Backend:** Ollama API (Gemma 3) ประมวลผลภาพพร้อมกฎระเบียบ (System Prompts) ผ่าน Vector Embedding (`nomic-embed-text`)
4. **Frontend UI:** Streamlit Dashboard จัดการ State Management และจำกัดการรีเฟรชหน้าเว็บที่ 1 FPS เพื่อรักษาเสถียรภาพการโต้ตอบของผู้ใช้

## ⚙️ การติดตั้งและรันระบบ (Setup & Installation)

### 1. ฝั่งฮาร์ดแวร์ (ESP32)
- เปิดโฟลเดอร์ `esp32_firmware` ด้วย PlatformIO
- เข้าไปที่ไฟล์ตั้งค่า และเปลี่ยน `SSID` กับ `PASSWORD` เป็นของเครือข่ายคุณ
- อัปโหลดโค้ดลงบอร์ด ESP32 และนำ IP Address ที่ได้มาใส่ในตัวแปร `CAMERAS_CONFIG` ในไฟล์ Python

### 2. ฝั่งซอฟต์แวร์ (Python & AI)
- ติดตั้ง Ollama และดาวน์โหลดโมเดลที่จำเป็น:
  ```bash
  ollama pull gemma3:latest
  ollama pull nomic-embed-text
- เปิดรัน Ollama server ทิ้งไว้เบื้องหลัง

- ติดตั้ง Python Dependencies:

pip install -r requirements.txt

- เริ่มต้นการทำงานของระบบ:

streamlit run app.py
