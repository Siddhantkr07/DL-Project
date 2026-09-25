# 🛡️ Sentinel AI – Cognitive Surveillance System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.1-ee4c2c?style=for-the-badge&logo=pytorch)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-black?style=for-the-badge)
![TensorRT](https://img.shields.io/badge/TensorRT-NVIDIA-76B900?style=for-the-badge&logo=nvidia)
![OpenAI CLIP](https://img.shields.io/badge/OpenAI-CLIP_ViT-white?style=for-the-badge&logo=openai)

Sentinel AI is a high-performance, multi-layered threat intelligence system designed for real-time video surveillance. Moving beyond standard bounding-box detection, this project implements a multi-stage pipeline (Skeleton Tracking + Object Detection + Zero-Shot Verification) to significantly reduce false positive alerts.

## 🚀 System Architecture

The pipeline is optimized for edge deployment (tested on RTX 4050 6GB) and operates in three distinct layers:

### 1. Dual-Model Tracking System (Primary Detection)
*   **YOLO11x-Pose (Skeleton Tracking):** Instead of relying on generic shapes, humans are detected and tracked using 17 specific body keypoints. This minimizes false positives where background objects are misclassified as people.
*   **YOLO11x (Object Detection):** Runs in parallel to detect potential threats such as weapons (knives/guns), vehicles, and hazards (fire/smoke).
*   **ByteTrack:** Advanced object tracking maintains consistent entity IDs across continuous frames.

### 2. Zero-Shot Verification Pipeline (Secondary Filter)
*   **OpenAI CLIP (Vision Transformer):** When YOLO detects a potential critical threat (e.g., a weapon), the specific region of interest (ROI) is cropped and passed to a ViT-B/32 CLIP model. 
*   **Cross-Examination:** CLIP acts as a secondary verifier by matching the visual crop against precise textual prompts ("a deadly knife" vs "a smartphone" vs "a pen"). If CLIP rejects the threat, the alert is discarded, effectively handling edge cases.

### 3. Threat Intelligence Engine (Temporal Analysis)
*   **Temporal Buffer:** A 5-frame rolling buffer ensures alerts are only triggered if a threat is consistently detected over consecutive frames, ignoring single-frame glitches.
*   **Pose-Based Fall Detection:** Fall detection relies on strict spatial geometry (checking if both ankles are visible, confident, and located on the same horizontal plane as the shoulders).
*   **Dynamic Alerting:** Alerts are classified into INFO, WARNING, and CRITICAL based on severity, crowd density, and proximity breaches.

---

## 💻 Tech Stack
*   **Core ML:** PyTorch, Ultralytics (YOLO11), HuggingFace Transformers (CLIP)
*   **Computer Vision:** OpenCV, ByteTrack
*   **Hardware Acceleration:** CUDA, NVIDIA TensorRT
*   **Backend & Dashboard:** Flask, Flask-SocketIO (WebSockets), JavaScript, Chart.js

---

## ⚙️ Hardware Acceleration (TensorRT)
To achieve maximum frames per second (FPS), the models can be compiled from `.pt` to NVIDIA TensorRT `.engine` formats. 
*Note: `.engine` files are highly hardware-specific and must be compiled directly on the host machine.*

---

## 🛠️ How to Run Locally

1. **Clone & Setup:**
   ```bash
   git clone https://github.com/Siddhantkr07/DL-Project.git
   cd DL-Project
   pip install -r requirements.txt
   ```
2. **Launch the Server:**
   ```bash
   python dashboard/app.py
   ```
3. **Open Dashboard:**
   Navigate to `http://127.0.0.1:5000` in your web browser.

---

## 👥 The Team
**VIT-AP University**
*   **Siddhant Kumar** (24BCA7078)
*   **Arava Kovid** (23BCE8596)
*   **S. Harsha Vardhan** (23BCE8166)
