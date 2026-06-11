"""
Harassment Detection System — Main Orchestrator (app.py)
=========================================================
Modules integrated:
  Module 1  — Device Input (camera + microphone)
  Module 2  — Real-Time Stream (threading)
  Module 3  — Video Processing (Haar Cascade face crop)
  Module 4  — Audio Processing (recording)
  Module 5  — Preprocessing (handled inside each model)
  Module 6  — Speech to Text
  Module 7  — Feature Extraction (handled inside each model)
  Module 8  — Model Inference (face emotion, voice emotion, text classifier)
  Module 9  — Decision Engine (multimodal fusion)
  Module 10 — Proof Storage (save image + audio on harassment)
  Module 11 — Database Logging (SQLite)

Run:
    python app.py
"""

import os
import time
import threading
import signal
import sys

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Module Imports ────────────────────────────────────────────────────────────
from modules.camera          import VideoStream, detect_and_crop_face
from modules.audio_record    import AudioRecorder
from modules.speech_to_text  import transcribe_audio
from modules.face_emotion    import predict_face_emotion
from modules.voice_emotion   import predict_voice_emotion
from modules.text_classifier import classify_text
from modules.decision_engine import make_decision
from modules.proof_storage   import save_proof
from database.database_helper import initialize_db, insert_record

# ── Configuration ─────────────────────────────────────────────────────────────
AUDIO_CHUNK_SECONDS   = 4       # Duration of each audio recording chunk
AUDIO_TEMP_PATH       = "temp/audio/live_chunk.wav"
FACE_PROCESS_INTERVAL = 2.0     # Seconds between face emotion evaluations
VIDEO_DISPLAY         = True    # Show OpenCV preview window

os.makedirs("temp/audio", exist_ok=True)
os.makedirs("proof/images", exist_ok=True)
os.makedirs("proof/audio", exist_ok=True)


# ── Shared state (thread-safe via locks) ──────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock             = threading.Lock()
        self.face_emotion     = "—"
        self.voice_emotion    = "—"
        self.text_prediction  = "—"
        self.transcribed_text = ""
        self.final_result     = "—"
        self.last_face_image  = None
        self.running          = True
        self.event_count      = 0


state = SharedState()


# ──────────────────────────────────────────────────────────────────────────────
# Thread 1: Video capture + face emotion
# ──────────────────────────────────────────────────────────────────────────────
def video_thread(stream: VideoStream):
    print("[VideoThread] Started.")
    last_process_time = 0

    while state.running:
        frame = stream.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        now = time.time()

        # Show preview (optional)
        if VIDEO_DISPLAY:
            import cv2
            preview = frame.copy()
            face, rect = detect_and_crop_face(preview)
            if rect:
                x, y, w, h = rect
                import cv2
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                with state.lock:
                    result = state.final_result
                color = (0, 0, 255) if result == "harassment" else \
                        (0, 165, 255) if result == "stress" else (0, 255, 0)
                cv2.putText(preview, result, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Harassment Detection System", preview)
            cv2.waitKey(1)

        # Run face emotion every FACE_PROCESS_INTERVAL seconds
        if now - last_process_time >= FACE_PROCESS_INTERVAL:
            face, _ = detect_and_crop_face(frame)
            if face is not None:
                emotion = predict_face_emotion(face)
                with state.lock:
                    state.face_emotion    = emotion
                    state.last_face_image = face.copy()
            last_process_time = now

    print("[VideoThread] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Thread 2: Audio capture + STT + text classification + voice emotion
# ──────────────────────────────────────────────────────────────────────────────
def audio_thread(recorder: AudioRecorder):
    print("[AudioThread] Started.")

    while state.running:
        # Record a chunk
        recorder.record_fixed(AUDIO_CHUNK_SECONDS, AUDIO_TEMP_PATH)

        # Parallel analysis: voice emotion + STT
        voice_result = predict_voice_emotion(AUDIO_TEMP_PATH)
        text         = transcribe_audio(AUDIO_TEMP_PATH)
        text_result  = classify_text(text) if text else "normal"

        with state.lock:
            state.voice_emotion    = voice_result
            state.transcribed_text = text
            state.text_prediction  = text_result

        # Trigger decision whenever we have fresh audio data
        fusion_step()

    print("[AudioThread] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Decision + logging step (called after each audio chunk)
# ──────────────────────────────────────────────────────────────────────────────
def fusion_step():
    with state.lock:
        face_e  = state.face_emotion
        voice_e = state.voice_emotion
        text_p  = state.text_prediction
        face_img = state.last_face_image
        state.event_count += 1
        event_id = state.event_count

    decision = make_decision(face_e, voice_e, text_p)

    proof_img   = None
    proof_audio = None

    if decision == "harassment":
        proof = save_proof(face_img, AUDIO_TEMP_PATH, event_id)
        proof_img   = proof["image_path"]
        proof_audio = proof["audio_path"]
        print(f"\n{'!'*60}")
        print(f"  ⚠ HARASSMENT DETECTED — Event #{event_id}")
        print(f"{'!'*60}\n")

    # Log to database
    insert_record(
        face_emotion    = face_e,
        speech_emotion  = voice_e,
        text_prediction = text_p,
        final_result    = decision,
        proof_image     = proof_img,
        proof_audio     = proof_audio,
    )

    with state.lock:
        state.final_result = decision

    print(
        f"[Event #{event_id}] "
        f"Face={face_e} | Voice={voice_e} | Text={text_p} | ► {decision}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────────────────────────────────────
def shutdown(sig=None, frame=None):
    print("\n[Main] Shutdown signal received. Stopping...")
    state.running = False


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Harassment Detection System — Starting")
    print("=" * 60)

    # Initialize database
    initialize_db()

    # Register CTRL+C handler
    signal.signal(signal.SIGINT, shutdown)

    # Initialize hardware
    stream   = VideoStream()
    recorder = AudioRecorder()

    stream.start()

    # Start threads
    t_video = threading.Thread(target=video_thread, args=(stream,), daemon=True)
    t_audio = threading.Thread(target=audio_thread, args=(recorder,), daemon=True)

    t_video.start()
    t_audio.start()

    print("[Main] System running. Press Ctrl+C to stop.")
    print("[Main] Open dashboard:  streamlit run dashboard/dashboard.py\n")

    try:
        while state.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()

    # Cleanup
    stream.stop()
    import cv2
    cv2.destroyAllWindows()
    print("[Main] System shut down cleanly.")


if __name__ == "__main__":
    main()
