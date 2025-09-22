# [Prototype] SDLock v0.1
---
## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [System Design](#system-design)
6. [Model Architecture](#model-architecture)
7. [Dataset](#dataset)
8. [Results and Evaluation](#results-and-evaluation)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)
---
## Introduction
<p align="justify">
This project presents a smart door lock system based on facial recognition. It uses MTCNN for face detection and InceptionResNetV1 for feature embedding. A custom anti-spoofing model named DualInputCNN enhances security by verifying face authenticity using both RGB and LBP image inputs.
</p>

## Features
- Real-time face detection and recognition
- Anti-spoofing to prevent fake face attacks
- Threshold-based Euclidean distance for identity verification
- Servo motor control for lock/unlock mechanism
- Integrated with Raspberry Pi and webcam
- Remote access via Flask API hosted on Google Colab

## Installation
- Clone this repository
- Download the DualInputCNN model from this repo: [AlvinOctaH/DualInputCNN](https://huggingface.co/AlvinOctaH/DualInputCNN)
- Mount your Google Drive in Colab
- Run colab_face_recognition.ipynb to start the Flask server
- Connect Raspberry Pi to the internet and run:
  - raspberry_pi_register_face.py to register users
  - raspberry_pi_face_recognition.py to perform recognition and unlock the door

## Usage
- Register face images via the Raspberry Pi script
- When a user is in front of the camera, the system:
  - Detects the face using MTCNN
  - Verifies liveness using anti-spoofing model
  - Extracts facial features with InceptionResNetV1
  - Compares against stored embeddings using Euclidean distance
  - Unlocks door via servo motor if identity is verified

## System Design
The system consists of hardware and software components designed to operate together through REST API communication. The lock mechanism is driven by an SG90 servo motor, controlled by a Raspberry Pi, and integrated with a webcam to capture facial images.
- The physical design includes a 3D-printed prototype
- All wiring and layout are integrated into a compact and functional housing

Prototype Views:
- Front, side, and bottom views of the device
<p align="center">
  <img width="480" height="360" src="https://raw.githubusercontent.com/AlvinOctaH/SDLockV0.1/main/assets/Imp1.jpg" alt="Front view of the device">
  <img width="480" height="360" src="https://raw.githubusercontent.com/AlvinOctaH/SDLockV0.1/main/assets/Imp2.jpg" alt="Bottom view of the device">
  <img width="480" height="360" src="https://raw.githubusercontent.com/AlvinOctaH/SDLockV0.1/main/assets/Imp3.jpg" alt="Side view of the device">
</p>

- Wiring schematic of the smart door lock
<p align="center">
  <img width="480" height="360" src="https://raw.githubusercontent.com/AlvinOctaH/SDLockV0.1/main/assets/Imp4.png" alt="Wiring schematic of the device">
</p>

- Exploded View, 3D Assembly, and Technical Drawing of the SDLock v0.1 System
<p align="center">
  <img width="480" height="360" src="https://raw.githubusercontent.com/AlvinOctaH/SDLockV0.1/main/assets/Imp5.png" alt="Exploded view of the device">
  <img width="480" height="360" src="https://raw.githubusercontent.com/AlvinOctaH/SDLockV0.1/main/assets/Imp6.jpg" alt="3D Assembly">
  <img width="480" height="360" src="https://raw.githubusercontent.com/AlvinOctaH/SDLockV0.1/main/assets/Imp7.jpg" alt="Technical Drawing">
</p>

## Model Architecture
- Anti-Spoofing: DualInputCNN model with RGB and LBP image branches
- Face Recognition: InceptionResNetV1 pretrained on VGGFace2
- Uses Euclidean distance to compare embeddings with a defined threshold (e.g., 0.8)

## Dataset
- Anti-spoofing: nguyenkhoa/antispoofing-3 (using only 10.000 data)
- Custom webcam images for finetuning anti-spoofing (190 data)
- Face recognition images registered manually via Raspberry Pi

## Results and Evaluation
- Face Detection: yields F1-score = 1.0
- Face Recognition: Best threshold at 0.8 yields F1-score = 0.87
- Anti-Spoofing Accuracy: 82.4% using "nguyenkhoa/antispoofing-3" dataset
- Security: FAR = 9.1%, FRR = 10.5%
- Performance: Average system response time ≈ 1.494 seconds

## Future Work
- Integrate Vision Transformer (ViT) with DINO pretraining for better spoofing resistance
- Add channel attention mechanism to enhance anti-spoofing
- Apply quantization and pruning for efficient edge deployment

## Contributing
Contributions are welcome. Please open issues and pull requests to collaborate or improve the system.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Author: Alvin Octa Hidayathullah
Email: alvinhidayatullah94@gmail.com
GitHub: https://github.com/AlvinOctaH
LinkedIn: www.linkedin.com/in/alvin-octa-hidayathullah
