# Wellness Watch (Harassment Detection System)

> Real-time multimodal AI surveillance and wellbeing monitoring system fusing computer vision (facial emotion), acoustic tone analysis, and BERT speech NLP into an integrated decision engine.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Overview

**Wellness Watch** is a multimodal AI system designed for real-time student wellbeing monitoring and harassment detection in academic, classroom, or workplace environments. By capturing concurrent video and audio feeds from standard laptop hardware, the system evaluates emotional affect through three distinct modalities:
1. **Visual**: Facial expression recognition (angry, sad, fearful, neutral, happy).
2. **Acoustic**: Speech prosody and voice tone emotion classification.
3. **Semantic**: Speech-to-text transcription paired with BERT toxic language classification.

A synchronized **13-module pipeline** coordinates capture, processing, fusion, encrypted incident logging, and real-time dashboard telemetry.

---

## 🚀 Key Features

- **Multimodal Tripartite Decision Engine (Module 9)**: Fuses visual cues, vocal stress, and transcript semantics with rule-based correlation to classify events as `NORMAL`, `STRESS`, or `HARASSMENT`.
- **Live Video Pipeline (Modules 1–3, 5, 8)**: OpenCV webcam frame acquisition, Haar Cascade face detection, facial emotion inference via CNN/ViT.
- **Concurrent Audio Pipeline (Modules 1–2, 4, 6–8)**: Non-blocking background microphone recording, noise reduction, faster speech transcription, and acoustic feature extraction.
- **Semantic Text Classification (Module 8)**: Hugging Face BERT transformer fine-tuned for abusive language and harassment detection on transcribed audio.
- **Proof & Evidence Vault (Module 10)**: Automated snapshot capture and audio segment archiving upon confirmed harassment flags.
- **Dual Database Persistence (Module 11)**: Native MongoDB support with zero-configuration SQLite fallback for standalone local environments.
- **Interactive Monitoring Dashboard (Modules 12–13)**: Streamlit operational control center displaying live timeline alerts, distress event logs, and status metrics.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.11 |
| **Computer Vision** | OpenCV (`cv2`), Haar Cascades, PIL |
| **Audio Processing** | SoundDevice, Librosa, SciPy, SpeechRecognition |
| **Deep Learning & NLP** | PyTorch (`torch`, `torchvision`, `torchaudio`), Hugging Face Transformers (BERT) |
| **Telemetry & UI** | Streamlit, Flask, Flask-CORS, Plyer (desktop notifications) |
| **Databases** | MongoDB (`pymongo`) / SQLite |

---

## 📂 Project Structure

```text
HarassmentDetectionSystem/
├── app.py                      # Master pipeline orchestrator & multi-threaded coordinator
├── main.py                     # Backward-compatible entrypoint alias
├── requirements.txt            # System dependencies
├── STRUCTURE.md                # In-depth architectural specification of Modules 1–13
├── dashboard/                  # Streamlit monitoring web dashboard
│   ├── app.py                  # Live alert and history viewer
│   └── ...
├── database/                   # Persistence layer (MongoDB / SQLite adapters)
│   └── database_helper.py
├── modules/                    # Core 13-stage multimodal processing pipeline
│   ├── device_input.py         # Module 1: Hardware webcam & microphone initialization
│   ├── realtime_stream.py      # Module 2: Threaded video/audio queue buffers
│   ├── video_processing.py     # Module 3: Face detection & crop
│   ├── audio_processing.py     # Module 4: Audio chunking & noise reduction
│   ├── preprocessing.py        # Module 5: Tensor normalization & tokenization
│   ├── speech_to_text.py       # Module 6: Audio transcription
│   ├── feature_extraction.py   # Module 7: MFCC and text feature extraction
│   ├── face_emotion.py         # Module 8a: Facial expression classifier
│   ├── voice_emotion.py        # Module 8b: Acoustic tone emotion model
│   ├── text_classifier.py      # Module 8c: BERT toxic text classifier
│   ├── multimodal_decision.py  # Module 9: Tri-factor decision & fusion engine
│   ├── proof_storage.py        # Module 10: Evidence capture (images + audio)
│   ├── database_module.py      # Module 11: Record logging
│   └── alert.py                # Module 12: Notification dispatcher
├── tests/                      # Verification and smoke test suites
│   ├── test_integration.py     # End-to-end pipeline verification (SQLite mode)
│   ├── test_device_input.py    # Hardware camera & mic detection test
│   ├── test_mic.py             # Audio input device validation
│   └── test_stream_module.py   # Queue throughput benchmark
└── proof/                      # Evidence directory (generated on confirmed alerts)
```

---

## ⚙️ Setup and Run Instructions

### 1. Prerequisites

- Python 3.11 installed
- Standard built-in laptop webcam and microphone (no external hardware required)
- MongoDB running locally (Optional: automatically falls back to SQLite)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Shagunbisht13/HarassmentDetectionSystem.git
cd HarassmentDetectionSystem

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Hardware & Integration

Run the diagnostic scripts to verify your webcam, microphone, and core pipeline:

```bash
# Verify camera & microphone detection
python test_device_input.py

# Run hardware-free integration test (validates decision engine & SQLite persistence)
python test_integration.py
```

---

## 💻 Usage

### 1. Start the Live Multimodal Monitoring Engine

```bash
python app.py
```

### 2. Launch the Streamlit Monitoring Dashboard

In a separate terminal:

```bash
streamlit run dashboard/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) to view real-time emotion telemetry, alert history, and captured proof artifacts.

---

## 📸 Screenshots

<!-- Replace placeholders with actual screenshots from your local run -->
| Live Detection Dashboard | Incident Evidence Vault |
| :---: | :---: |
| `![Dashboard Placeholder](https://via.placeholder.com/450x300.png?text=Wellness+Watch+Dashboard)`<br>*(Real-time emotion stream & alert banner)* | `![Evidence Vault Placeholder](https://via.placeholder.com/450x300.png?text=Proof+Storage+Vault)`<br>*(Encrypted incident logs and timestamped proof)* |

---

## ⚠️ Limitations

- **Hardware Latency**: Simultaneous real-time execution of video processing, audio STT, and BERT NLP inference requires a moderately modern CPU/GPU for sub-second alert response times.
- **Lighting & Acoustic Conditions**: Haar Cascade face detection accuracy degrades in low-light environments, and heavy background ambient noise may impede microphone transcription.
- **Benchmark Metrics**: [TODO: add test accuracy / validation F1 score for emotion models once benchmarked on standard datasets].

---

## 🔮 Future Improvements

- [ ] Transition face detection from Haar Cascades to **MediaPipe / YOLOv8-Face** for higher precision at steep angles.
- [ ] Export BERT and emotion classifiers to **ONNX Runtime / TensorRT** to minimize inference latency on edge devices.
- [ ] Implement an automated email / SMS escalation webhook for rapid intervention.
