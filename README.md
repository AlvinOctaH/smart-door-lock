# 🚪 Smart Door Lock — Face Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Raspberry%20Pi-4B-red?logo=raspberrypi" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/Architecture-Client--Server-green" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <strong>Design and Development of a Smart Door Lock System Based on Face Recognition Using MTCNN-InceptionResNet</strong><br/>
  <em>Undergraduate Thesis — Robotics and Artificial Intelligence Engineering, Universitas Airlangga (2025)</em>
</p>

---

## 📖 Overview

This project presents a **face recognition-based smart door lock** system that replaces traditional key-based access with biometric authentication. The system integrates:

- **MTCNN** for robust multi-scale face detection under varying lighting conditions
- **InceptionResNetV1** (via FaceNet-PyTorch) for face embedding and recognition
- **DualInputCNN** — a custom anti-spoofing model combining RGB image features and Local Binary Pattern (LBP) descriptors

The system runs on a **Raspberry Pi 4 Model B** using a **client-server architecture**, offloading inference to a Google Colab-based server to overcome local computational limits.

> 📄 **Paper/Thesis**: [[arXiv link — coming soon]](https://arxiv.org) | [Universitas Airlangga Repository](https://repository.unair.ac.id)

---

## 🏗️ System Architecture

```
┌─────────────────────────┐          ┌──────────────────────────────┐
│     Raspberry Pi 4B     │          │    Server (Google Colab)     │
│                         │          │                              │
│  Webcam (Logitech C922) │          │  ┌─────────────────────────┐ │
│         ↓               │  HTTP    │  │ 1. MTCNN Face Detection  │ │
│  Capture Frame          │ ───────► │  │ 2. Anti-Spoofing (CNN)   │ │
│         ↓               │          │  │ 3. InceptionResNetV1     │ │
│  REST API Request       │ ◄─────── │  │    (Face Recognition)    │ │
│         ↓               │  JSON    │  └─────────────────────────┘ │
│  Servo Motor (SG90)     │          └──────────────────────────────┘
│  → Unlock Door          │
└─────────────────────────┘
```

---

## 📊 Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| **Face Detection (MTCNN)** | Precision / Recall / F1-score | **1.0 / 1.0 / 1.0** (all 4 lighting conditions) |
| **Face Recognition (InceptionResNetV1)** | Average FAR | 9.1% (threshold = 0.8) |
| **Face Recognition (InceptionResNetV1)** | Average FRR | 10.5% (threshold = 0.8) |
| **Anti-Spoofing (DualInputCNN)** | Accuracy on test dataset | 82.4% |
| **System Response Time** | Avg. end-to-end latency | 1.494 seconds @ 23.49 Mbps |

---

## 🔬 Models

### 1. Face Detection — MTCNN
Multi-Task Cascaded Convolutional Neural Network with three stages (P-Net, R-Net, O-Net) for face detection and facial landmark alignment. Selected over BlazeFace, SSD, and YOLO for its superior accuracy in challenging lighting conditions.

### 2. Face Recognition — InceptionResNetV1
Pre-trained model from the [facenet-pytorch](https://github.com/timesler/facenet-pytorch) library (VGGFace2 weights). Produces 512-dimensional face embeddings, compared via Euclidean distance at threshold `0.8`.

### 3. Anti-Spoofing — DualInputCNN
A custom CNN model with dual inputs:
- **RGB branch**: captures color and texture features
- **LBP branch**: extracts local texture descriptors for liveness detection

Trained on `nguyenkhoa/antispoofing-3` dataset (~104,000 images) and fine-tuned on primary webcam data.

**DualInputCNN Architecture:**

| Layer | Details |
|-------|---------|
| Input (RGB) | 160×160×3 |
| Input (LBP) | 160×160×1 |
| Conv blocks | 3× (Conv2D + BatchNorm + MaxPool) |
| Fusion | Concatenate + Dense layers |
| Output | Sigmoid (real / spoof) |

Training parameters: Adam optimizer, lr=0.0001, dropout=0.3, max 200 epochs, patience=20.

---

## 🛠️ Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **Edge Device** | Raspberry Pi 4 Model B (4GB RAM) |
| **Camera** | Logitech C922 Pro HD Stream Webcam (1080p@30fps) |
| **Actuator** | TowerPro SG90 Servo Motor |
| **Connectivity** | LAN Cat 6 / WiFi |
| **Storage** | MicroSDHC |

---

## 💻 Software Requirements

```
Python 3.8+
torch >= 1.9.0
facenet-pytorch
opencv-python
numpy
flask            # for REST API server
RPi.GPIO         # for Raspberry Pi GPIO control
```

---

## 📂 Repository Structure

```
SDLockV0.1/
│
├── src/
│   ├── train_model_anti_spoofing.ipynb      # Anti-spoofing model training (Google Colab)
│   ├── colab_face_recognition.ipynb         # Main inference server (Google Colab)
│   ├── raspberry_pi_register_face.py        # Face enrollment script (Raspberry Pi)
│   └── raspberry_pi_face_recognition.py    # Main door lock script (Raspberry Pi)
│
├── models/
│   └── dual_input_cnn_finetuned.pth         # Trained anti-spoofing model weights
│
├── docs/
│   ├── system_architecture.png
│   └── wiring_diagram.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AlvinOctaH/SDLockV0.1.git
cd SDLockV0.1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up the inference server (Google Colab)
Open `src/colab_face_recognition.ipynb` in Google Colab, run all cells, and copy the **ngrok public URL** for the API endpoint.

### 4. Register a face
```bash
# On Raspberry Pi
python src/raspberry_pi_register_face.py --name "YourName"
```

### 5. Run the door lock system
```bash
# On Raspberry Pi
python src/raspberry_pi_face_recognition.py --server-url "https://YOUR-NGROK-URL"
```

---

## 🧪 Evaluation

The system was evaluated across **four lighting conditions**:
- Dim indoor (10–50 lux)
- Normal indoor (100–300 lux)
- Outdoor daylight (1,000–2,000 lux)
- Late afternoon (640–1,000 lux)

Additional tests include:
- Face recognition with accessories (glasses, masks, hats)
- Spoofing resistance (printed photos & smartphone screen)
- Partial face occlusion detection
- Black box feature testing

---

## ⚠️ Known Limitations

- Anti-spoofing fails on high-quality face images displayed on smartphone screens (70% detection rate in real-world testing)
- Face recognition accuracy degrades when facial features are occluded (e.g., sunglasses, full mask)
- System depends on stable internet connection for server inference; higher latency at lower speeds

---

## 🔮 Future Work

- [ ] Integrate pre-trained anti-spoofing models: **MobileNetV3-based CDCN**, **DINO-ViT**, or **CDCN++**
- [ ] Add spatio-temporal features to improve liveness detection against video replay attacks
- [ ] Evaluate **LVFace** (ViT-based Progressive Cluster Optimization) for face recognition
- [ ] Deploy fully local inference on **Raspberry Pi 5 + AI Kit** or **NVIDIA Jetson Orin Nano**
- [ ] Expand test dataset with more accessory and pose variations

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{hidayathullah2025smartdoorlock,
  author    = {Alvin Octa Hidayathullah},
  title     = {Design and Development of a Smart Door Lock System Based on Face Recognition Using MTCNN-InceptionResNet},
  year      = {2025},
  school    = {Universitas Airlangga},
  note      = {Undergraduate Thesis},
  url       = {https://github.com/AlvinOctaH/SDLockV0.1}
}
```

---

## 👤 Author

**Alvin Octa Hidayathullah**  
S1 Teknik Robotika dan Kecerdasan Buatan  
Fakultas Teknologi Maju dan Multidisiplin, Universitas Airlangga  
📧 *(add your email here)*  
🔗 [LinkedIn](https://linkedin.com) | [GitHub](https://github.com/AlvinOctaH)

---

## 🙏 Acknowledgements

- **Amila Sofiah, S.T., M.T.** — Supervisor I
- **Dr. Maryamah, S.Kom.** — Supervisor II
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) by Timothy Esler
- Dataset: [nguyenkhoa/antispoofing-3](https://huggingface.co/datasets/nguyenkhoa/antispoofing-3) on Hugging Face

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
