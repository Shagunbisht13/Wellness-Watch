# Harassment Detection System

## Overview

The **Harassment Detection System** is an end‑to‑end multimodal pipeline that monitors live video and audio streams, detects potential harassment, and provides real‑time alerts via a web dashboard.

- **Video branch** – captures webcam frames, detects faces, extracts facial emotion using a CNN/ViT model.
- **Audio branch** – records microphone chunks, performs speech‑to‑text, extracts speech emotion, and runs a BERT‑based harassment classifier on the transcribed text.
- **Fusion** – combines facial emotion, speech emotion, and text prediction to make a final decision, stores proof evidence, logs the event in a database, and triggers an alert on the dashboard.

The system consists of 13 modular Python components that run concurrently in separate threads.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Core language** | Python 3.11 |
| **Video capture & processing** | OpenCV, Haar Cascades |
| **Audio capture & processing** | sounddevice, librosa, SpeechRecognition |
| **Feature extraction** | NumPy, SciPy, PyTorch (torch, torchvision, torchaudio) |
| **NLP / Classification** | Transformers (BERT), HuggingFace 🤗 hub |
| **Web dashboard** | Streamlit |
| **Database** | MongoDB (via `pymongo`) |
| **REST / Server** | Flask (used for alert API) |
| **Utilities** | pandas, matplotlib, Pillow, nltk, plyer |

---

## Repository Structure

```
HarassmentDetectionSystem/
├─ app.py                 # Main orchestrator – starts pipelines & handles shutdown
├─ main.py                # Simple entry point (currently unused)
├─ requirements.txt       # Python dependencies
├─ modules/               # All pipeline modules
│   ├─ realtime_stream.py       # Stream manager (video & audio queues)
│   ├─ video_processing.py      # Face detection & cropping
│   ├─ audio_processing.py      # Audio cleaning & feature extraction
│   ├─ audio_record.py          # Legacy backward-compatible audio recorder shim
│   ├─ preprocessing.py         # Face & text preprocessing utilities
│   ├─ speech_to_text.py        # STT using SpeechRecognition
│   ├─ feature_extraction.py    # Audio, image, text feature extractors
│   ├─ pretrained_models.py     # Emotion & harassment classifiers
│   ├─ multimodal_decision.py   # Fusion logic
│   ├─ proof_storage.py         # Saves image/audio proof files
│   ├─ database_module.py       # MongoDB helper functions
│   └─ alert.py                 # Sends alert information to the dashboard
├─ dashboard/            # Streamlit dashboard source
│   └─ dashboard.py
├─ Temp/                 # Temporary audio chunks (auto‑created at runtime)
├─ proof/                # Stored proof images/audio (auto‑created at runtime)
└─ README.md             # <-- This file
```

---

## Setup & Installation

1. **Clone the repository** (already done in your workspace).
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # on Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > The installation may take several minutes as it pulls large ML packages.
4. **Download the required pretrained models** (the code will automatically download them on first run via `torch.hub` / `transformers`).

---

## Running the System (Demo Steps)

1. **Start the backend pipeline**:
   ```bash
   python app.py
   ```
   - This launches two daemon threads – one for video processing and one for audio/text processing.
   - The system will create the temporary folders (`Temp/`, `proof/`) if they do not exist.
2. **Launch the dashboard** (in a **separate terminal**):
   ```bash
   streamlit run dashboard/dashboard.py
   ```
   - Streamlit will start on `http://localhost:8501` and automatically open in your default browser.
3. **Interact with the system**:
   - Position your webcam and speak near the microphone.
   - The dashboard will display a live video preview, current emotion predictions, transcribed text, and the final decision ("harassment", "stress", or "normal").
   - When harassment is detected, a proof image/audio pair is saved under `proof/` and a new row is inserted into the MongoDB database.
4. **Stop the system**:
   - Press `Ctrl+C` in the terminal running `app.py`.
   - The program will cleanly shut down the camera windows and stop all threads.

---

## Database & Proof Storage

- The system uses **MongoDB** (default connection string is `mongodb://localhost:27017`). Ensure MongoDB is running locally or update the connection details in `modules/database_module.py`.
- Proof images are saved under `proof/images/` and audio under `proof/audio/`. The dashboard reads these folders to display the evidence for each harassment event.

---

## Extending / Customising

- **Add new models** – Replace the functions in `modules/pretrained_models.py` with your own Torch or TensorFlow models and adjust the input preprocessing accordingly.
- **Change the decision logic** – Modify `modules/multimodal_decision.py` to weight emotions differently or incorporate additional sensors.
- **Deploy the dashboard** – Streamlit can be containerised (Docker) or deployed to a cloud VM; just expose port `8501` and point the frontend to the backend’s API.

---

## Troubleshooting

- **`streamlit` not found** – Ensure the virtual environment is activated and `pip install streamlit` succeeded.
- **Webcam not opening** – Verify that no other application is using the camera and that OpenCV can access it (`cv2.VideoCapture(0)`).
- **MongoDB connection errors** – Start a local MongoDB instance or edit the connection string in `modules/database_module.py`.
- **Audio device errors** – Make sure a microphone is connected and that the `sounddevice` library can list devices (`sounddevice.query_devices()`).

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgements

- OpenCV community for face detection utilities.
- HuggingFace Transformers for the BERT models.
- Streamlit team for the rapid dashboard development.
- The many open‑source libraries listed in `requirements.txt`.

---

*Happy hacking! 🎉*
