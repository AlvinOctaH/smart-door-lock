# 🚪 Smart Door Lock — Face Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Raspberry%20Pi-4B-red?logo=raspberrypi" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/Server-Google%20Colab-yellow?logo=googlecolab" />
  <img src="https://img.shields.io/badge/API-Flask%20%2B%20ngrok-lightblue?logo=flask" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <strong>Design and Development of a Smart Door Lock System Based on Face Recognition Using MTCNN-InceptionResNet</strong><br/>
  <em>Undergraduate Thesis — Robotics and Artificial Intelligence Engineering, Universitas Airlangga (2025)</em>
</p>

---

## 📖 Overview

This project presents a **face recognition-based smart door lock** system that replaces traditional key-based access with biometric authentication. The system uses a **client-server architecture**:

- **Edge (Raspberry Pi 4B)**: Captures frames via webcam, sends them to the inference server via REST API, and controls the servo motor lock
- **Server (Google Colab)**: Runs MTCNN face detection, DualInputCNN anti-spoofing, and InceptionResNetV1 face recognition, exposed via Flask + ngrok

> 📄 **Paper/Thesis**: [[arXiv — coming soon]](https://arxiv.org) | [Universitas Airlangga Repository](https://repository.unair.ac.id)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────┐         ┌─────────────────────────────────────┐
│        Raspberry Pi 4B          │         │       Server (Google Colab)         │
│                                 │         │                                     │
│  Logitech C922 Webcam           │         │  ┌──────────────────────────────┐  │
│         │                       │  HTTP   │  │  POST /register_face         │  │
│  Capture frame.jpg              │ ──────► │  │  POST /face_recognition      │  │
│         │                       │         │  └──────────────────────────────┘  │
│  requests.post(ngrok_url)       │         │                │                   │
│         │                       │ ◄─────  │  1. MTCNN — Face Detection         │
│  Parse JSON response            │  JSON   │  2. DualInputCNN — Anti-Spoofing   │
│         │                       │         │  3. InceptionResNetV1 — Embedding  │
│  if success:                    │         │  4. Euclidean Distance Matching    │
│    pigpio → Servo SG90 → 90°   │         │  → JSON { status, name }           │
│    time.sleep(5) → back to 0°  │         └─────────────────────────────────────┘
└─────────────────────────────────┘
         ↕
    config.txt
  (ngrok public URL)
```

---

## 📊 Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| **Face Detection (MTCNN)** | Precision / Recall / F1-score | **1.0 / 1.0 / 1.0** (4 lighting conditions) |
| **Face Recognition (InceptionResNetV1)** | False Acceptance Rate (FAR) | 9.1% @ threshold 0.8 |
| **Face Recognition (InceptionResNetV1)** | False Rejection Rate (FRR) | 10.5% @ threshold 0.8 |
| **Anti-Spoofing (DualInputCNN)** | Accuracy on test dataset | 82.4% |
| **System Response Time** | Avg. end-to-end latency | 1.494s @ 23.49 Mbps |

---

## 🧠 Models

### 1. Face Detection — MTCNN
Multi-Task Cascaded CNN (P-Net → R-Net → O-Net) from `facenet-pytorch`. Handles varying lighting (10–2000 lux), partial occlusion, and accessories.

```python
from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=240, margin=0, device=device)
boxes, _ = mtcnn.detect(image)  # Returns bounding boxes
```

### 2. Face Recognition — InceptionResNetV1
Pre-trained on VGGFace2 (3M+ images, 9k+ identities). Produces 512-dim embeddings compared via Euclidean distance.

```python
from facenet_pytorch import InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
embedding = resnet(face_tensor.unsqueeze(0))  # shape: [1, 512]
```

Best threshold: **0.8** (FAR 9.1%, FRR 10.5%, avg F1-score 0.87)

### 3. Anti-Spoofing — DualInputCNN
Custom CNN with **dual parallel branches** fused before classification:

```
RGB Input (3×240×240)  →  Conv64 → Conv128 → Conv256 → Dropout(0.2)  ─┐
                                                                         ├→ Concat(512ch) → Conv512 → Conv512 → AdaptivePool(7×7) → FC512 → FC128 → FC2
LBP Input (1×240×240)  →  Conv64 → Conv128 → Conv256 → Dropout(0.2)  ─┘
```

LBP is computed with `skimage.feature.local_binary_pattern(gray, P=8, R=1, method='uniform')`.

**Training configuration:**

| Stage | Dataset | lr | Max Epochs | Patience | Dropout |
|-------|---------|-----|------------|---------|---------|
| Initial training | nguyenkhoa/antispoofing-3 (8,000 imgs) | 1e-4 | 200 | 20 | 0.3 |
| Fine-tuning | Primary webcam data | 1e-5 | 100 | 10 | 0.3 |
| LR Scheduler | StepLR(gamma=0.5, step=100) for initial; StepLR(gamma=0.5, step=30) for fine-tuning | — | — | — | — |

---

## 🛠️ Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Edge device | Raspberry Pi 4 Model B (4GB RAM) |
| Camera | Logitech C922 Pro HD Stream (1080p@30fps) |
| Actuator | TowerPro SG90 Servo Motor (GPIO18 / Pin 12) |
| Connectivity | LAN Cat 6 or WiFi (recommended ≥ 20 Mbps) |

**Servo wiring (SG90):**
- Grey → GND
- Red → VCC (4.8V–7.2V)
- Orange → GPIO18 (PWM signal via `pigpio`)

Pulse width mapping: `pw = 500 + ((180 - angle) / 180.0) * 2000` µs (inverted direction)

---

## 💻 Software Requirements

```
# Server (Google Colab)
facenet-pytorch
flask
pyngrok
scikit-image
torch >= 1.9.0
torchvision
pillow
numpy
grad-cam          # for Grad-CAM visualization during training

# Raspberry Pi
opencv-python
requests
pigpio            # Note: requires 'sudo pigpiod' before running
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure

```
SDLockV0.1/
│
├── src/
│   ├── train_model_anti_spoofing.ipynb      # DualInputCNN training + fine-tuning (Colab)
│   ├── colab_face_recognition.ipynb         # Flask inference server (Colab + ngrok)
│   ├── raspberry_pi_register_face.py        # Face enrollment client (Raspberry Pi)
│   └── raspberry_pi_face_recognition.py    # Main door lock loop (Raspberry Pi)
│
├── models/
│   └── dual_input_cnn_finetuned.pth         # Fine-tuned anti-spoofing weights (Google Drive)
│
├── config.txt                               # Stores ngrok public URL (update each Colab session)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Step 1 — Clone the repo
```bash
git clone https://github.com/AlvinOctaH/SDLockV0.1.git
cd SDLockV0.1
```

### Step 2 — Start the inference server (Google Colab)

1. Open `src/colab_face_recognition.ipynb` in Google Colab
2. Mount Google Drive — model weights (`dual_input_cnn_finetuned.pth`) and face database (`face_database.pkl`) are loaded from there
3. Set your ngrok auth token in the config cell:
   ```python
   from pyngrok import conf
   conf.get_default().auth_token = "YOUR_NGROK_TOKEN"
   ```
4. Run all cells — the public ngrok URL will be printed as `* Ngrok URL: https://xxxx.ngrok-free.app`
5. Copy the URL into `config.txt` on your Raspberry Pi:
   ```
   https://xxxx-xx-xx-xxx-xx.ngrok-free.app
   ```

> ⚠️ The ngrok URL changes every Colab session. Remember to update `config.txt` each time.

### Step 3 — Start the pigpio daemon (Raspberry Pi)
```bash
sudo pigpiod
```

### Step 4 — Register a face
```bash
python src/raspberry_pi_register_face.py
# A camera window will open (480×360)
# Press 's' to capture your photo
# Enter your name when prompted
# The face embedding is stored in face_database.pkl on Google Drive
```

### Step 5 — Run the door lock
```bash
python src/raspberry_pi_face_recognition.py
# Live camera feed shown in a 480×360 window
# Each frame is sent to the server for recognition
# On success: servo rotates to 90° → waits 5s → returns to 0°
# Press 'q' to quit
```

---

## 🔌 API Reference

Both endpoints are served by the Flask server in `colab_face_recognition.ipynb`.

### `POST /register_face`
Registers a new face embedding into the pickle database.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | file | Face image (JPEG/PNG) |
| `name` | string | Name label for this face |

**Success:**
```json
{ "status": "success", "message": "Face registered for Alvin" }
```
**No face detected:**
```json
{ "status": "failed", "message": "No face detected" }
```

---

### `POST /face_recognition`
Runs the full pipeline: detection → anti-spoofing → recognition → door control signal.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | file | Webcam frame (JPEG) |

**Access granted:**
```json
{ "status": "success", "name": "Alvin" }
```
**Spoof detected:**
```json
{ "status": "failed", "message": "Spoof detected" }
```
**Unknown face:**
```json
{ "status": "failed", "message": "Face not recognized" }
```

---

## ⚠️ Known Limitations

- **Anti-spoofing**: Real-world detection rate is 70% (7/10). Fails on high-quality face photos displayed at full focus on a smartphone screen. Works better when the spoofed image is slightly blurred.
- **Accessories**: Recognition drops significantly with sunglasses or masks covering eyes and mouth (F1-score ~0.4–0.5 for those cases).
- **Internet dependency**: Response time scales with bandwidth — 1.494s @ 23.49 Mbps, 1.825s @ 15.31 Mbps.
- **ngrok URL lifecycle**: URL expires on Colab disconnect; `config.txt` must be updated manually each session.

---

## 🔮 Future Work

- [ ] Replace DualInputCNN with pre-trained models: **MobileNetV3-CDCN**, **DINO-ViT**, or **CDCN++**
- [ ] Add spatio-temporal features for video replay attack detection
- [ ] Evaluate **LVFace** (ViT + Progressive Cluster Optimization) for face recognition
- [ ] Deploy full local inference on **Raspberry Pi 5 + AI Kit** or **NVIDIA Jetson Orin Nano**
- [ ] Use a persistent server to eliminate the ngrok URL rotation problem
- [ ] Expand evaluation dataset for varied poses and accessories

---

## 📚 Citation

```bibtex
@misc{hidayathullah2025smartdoorlock,
  author = {Alvin Octa Hidayathullah},
  title  = {Design and Development of a Smart Door Lock System Based on Face Recognition Using MTCNN-InceptionResNet},
  year   = {2025},
  school = {Universitas Airlangga},
  note   = {Undergraduate Thesis},
  url    = {https://github.com/AlvinOctaH/SDLockV0.1}
}
```

---

## 👤 Author

**Alvin Octa Hidayathullah**  
S1 Teknik Robotika dan Kecerdasan Buatan  
Fakultas Teknologi Maju dan Multidisiplin, Universitas Airlangga  
🔗 [GitHub](https://github.com/AlvinOctaH/SDLockV0.1)

---

## 🙏 Acknowledgements

- **Amila Sofiah, S.T., M.T.** — Supervisor I
- **Dr. Maryamah, S.Kom.** — Supervisor II
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) by Timothy Esler
- Dataset: [nguyenkhoa/antispoofing-3](https://huggingface.co/datasets/nguyenkhoa/antispoofing-3)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
