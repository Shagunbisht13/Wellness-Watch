"""
Harassment Detection System — Main Orchestrator (app.py / main entry)
=====================================================================
End-to-end workflow (exact pipeline order):

  1.  Webcam captures live video          (Module 1 — Device Input)
  2.  OpenCV extracts frame               (Module 2 — Real-Time Stream)
  3.  Haar Cascade detects face           (Module 3 — Video Processing)
  4.  Face image preprocessed             (Module 5 — Preprocessing)
  5.  Microphone records speech (parallel)(Modules 1–2 + 4 — Audio Processing)
  6.  Speech converted to text            (Module 6 — Speech-to-Text)
  7.  BERT analyzes sentence meaning      (Modules 7–8 — Features + Text BERT)
  8.  CNN/ViT detects facial emotion      (Module 8 — Face Emotion)
  9.  Audio model detects voice tone      (Module 8 — Speech Emotion)
 10. Decision module combines results     (Module 9 — Multimodal Decision)
 11. If harassment → store proof          (Module 10 — Proof Storage)
 12. Result saved to database             (Module 11 — Database)
 13. Alert shown on dashboard             (Modules 12–13 — Alert + Dashboard)

Run:
    python app.py
    streamlit run dashboard/dashboard.py
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

# ── Modules 1–13 ──────────────────────────────────────────────────────────────
from modules.realtime_stream import StreamManager                    # 1+2
from modules.video_processing import detect_and_crop_face           # 3
from modules.audio_processing import process_audio_chunk            # 4
from modules.preprocessing import preprocess_face, preprocess_text  # 5
from modules.speech_to_text import transcribe_audio                 # 6
from modules.feature_extraction import (                            # 7
    extract_audio_features,
    extract_image_features,
    extract_text_features,
)
from modules.pretrained_models import (                             # 8
    classify_text,
    predict_face_emotion,
    predict_speech_emotion,
)
from modules.multimodal_decision import make_decision               # 9
from modules.proof_storage import save_proof                        # 10
from modules.database_module import initialize_db, insert_record    # 11
from modules.alert import trigger_alert                             # 12
# Module 13 — dashboard runs separately: streamlit run dashboard/dashboard.py

# ── Configuration ─────────────────────────────────────────────────────────────
AUDIO_CHUNK_SECONDS = 4.0
AUDIO_TEMP_PATH = os.path.join("Temp", "audio", "live_chunk.wav")
SAMPLE_RATE = 16000
FRAME_CAPTURE_FPS = 18
VIDEO_DISPLAY = True

os.makedirs(os.path.join("Temp", "audio"), exist_ok=True)
os.makedirs(os.path.join("proof", "images"), exist_ok=True)
os.makedirs(os.path.join("proof", "audio"), exist_ok=True)


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.face_emotion = "—"
        self.voice_emotion = "—"
        self.text_prediction = "—"
        self.transcribed_text = ""
        self.final_result = "—"
        self.last_face_image = None
        self.running = True
        self.event_count = 0


state = SharedState()


def video_pipeline(stream: StreamManager):
    """
    Workflow steps 1–4 + 8 (face branch):
      frame → Haar face crop → preprocess → face emotion
    """
    print("[VideoPipeline] Started.")
    while state.running:
        # Steps 1–2: live frame from stream queue (~1 FPS processed)
        frame = stream.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # Step 3: Haar Cascade face detection + crop
        face, rect = detect_and_crop_face(frame)

        if VIDEO_DISPLAY:
            import cv2
            preview = frame.copy()
            if rect:
                x, y, w, h = rect
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                with state.lock:
                    result = state.final_result
                color = (
                    (0, 0, 255)
                    if result == "harassment"
                    else (0, 165, 255)
                    if result == "stress"
                    else (0, 255, 0)
                )
                cv2.putText(
                    preview,
                    result,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            cv2.imshow("Harassment Detection System", preview)
            cv2.waitKey(1)

        if face is None:
            continue

        # Step 4: face preprocessing (also used inside Module 8)
        _ = preprocess_face(face, size=(224, 224))
        # Step 7 (image features) + Step 8 (face emotion CNN/ViT)
        _ = extract_image_features(face)
        emotion = predict_face_emotion(face)

        with state.lock:
            state.face_emotion = emotion
            state.last_face_image = face.copy()

    print("[VideoPipeline] Stopped.")


def audio_pipeline(stream: StreamManager):
    """
    Workflow steps 5–7 + 9–13 (audio/text branch + fusion):
      chunk → process → STT → BERT text → speech emotion → decision → proof/DB/alert
    """
    print("[AudioPipeline] Started.")
    while state.running:
        # Step 5: microphone chunk (parallel with video)
        chunk = stream.get_audio_chunk()
        if chunk is None:
            time.sleep(0.05)
            continue

        # Step: Module 4 audio processing → clean WAV
        process_audio_chunk(
            chunk,
            sample_rate=stream.sample_rate,
            output_path=AUDIO_TEMP_PATH,
        )

        # Step 9 path: speech emotion from cleaned audio
        voice_result = predict_speech_emotion(AUDIO_TEMP_PATH)
        _ = extract_audio_features(AUDIO_TEMP_PATH)

        # Step 6: speech → text
        text = transcribe_audio(AUDIO_TEMP_PATH)
        clean_text = preprocess_text(text) if text else ""

        # Steps 7–8: BERT features + BERT harassment classifier
        if clean_text:
            _ = extract_text_features(clean_text)
        text_result = classify_text(text) if text else "normal"

        with state.lock:
            state.voice_emotion = voice_result
            state.transcribed_text = text
            state.text_prediction = text_result

        # Steps 10–13
        fusion_step()

    print("[AudioPipeline] Stopped.")


def fusion_step():
    """Steps 10–13: decision → proof → database → alert."""
    with state.lock:
        face_e = state.face_emotion
        voice_e = state.voice_emotion
        text_p = state.text_prediction
        face_img = state.last_face_image
        state.event_count += 1
        event_id = state.event_count

    # Step 10: multimodal decision
    decision = make_decision(face_e, voice_e, text_p)

    proof_img = None
    proof_audio = None

    # Step 11: store proof only on harassment
    if decision == "harassment":
        proof = save_proof(face_img, AUDIO_TEMP_PATH, event_id)
        proof_img = proof["image_path"]
        proof_audio = proof["audio_path"]

    # Step 12: database history
    insert_record(
        face_emotion=face_e,
        speech_emotion=voice_e,
        text_prediction=text_p,
        final_result=decision,
        proof_image=proof_img,
        proof_audio=proof_audio,
    )

    # Step 13: alert (dashboard reads DB + alert log)
    trigger_alert(
        final_result=decision,
        face_emotion=face_e,
        voice_emotion=voice_e,
        text_prediction=text_p,
        event_id=event_id,
    )

    with state.lock:
        state.final_result = decision

    print(
        f"[Event #{event_id}] "
        f"Face={face_e} | Voice={voice_e} | Text={text_p} | -> {decision}"
    )


def shutdown(sig=None, frame=None):
    print("\n[Main] Shutdown signal received. Stopping...")
    state.running = False


def main():
    print("=" * 60)
    print("  Harassment Detection System — 13-Module Pipeline")
    print("=" * 60)

    initialize_db()
    signal.signal(signal.SIGINT, shutdown)

    # Modules 1 + 2: device input + real-time stream threads
    stream = StreamManager(
        frame_fps=FRAME_CAPTURE_FPS,
        audio_chunk_duration=AUDIO_CHUNK_SECONDS,
        sample_rate=SAMPLE_RATE,
    )
    stream.start()

    t_video = threading.Thread(target=video_pipeline, args=(stream,), daemon=True)
    t_audio = threading.Thread(target=audio_pipeline, args=(stream,), daemon=True)
    t_video.start()
    t_audio.start()

    print("[Main] System running. Press Ctrl+C to stop.")
    print("[Main] Dashboard (Module 13): streamlit run dashboard/dashboard.py\n")

    try:
        while state.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()

    stream.stop()
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("[Main] System shut down cleanly.")


if __name__ == "__main__":
    main()
