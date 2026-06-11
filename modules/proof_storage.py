"""
Module 10: Proof Storage
- Saves evidence (face image + audio clip) when harassment is detected.
- Creates timestamped filenames for easy reference.
- Directories: proof/images/ and proof/audio/
"""

import os
import shutil
import cv2
import numpy as np
from datetime import datetime


PROOF_IMAGES_DIR = "proof/images"
PROOF_AUDIO_DIR  = "proof/audio"


def _ensure_dirs():
    os.makedirs(PROOF_IMAGES_DIR, exist_ok=True)
    os.makedirs(PROOF_AUDIO_DIR, exist_ok=True)


def save_proof(
    face_image: np.ndarray | None,
    audio_wav_path: str | None,
    event_id: int | None = None
) -> dict:
    """
    Save evidence of a harassment event.

    Args:
        face_image:     BGR face image numpy array (or None).
        audio_wav_path: Path to the audio chunk WAV file (or None).
        event_id:       Optional integer ID for the event.

    Returns:
        Dict with keys 'image_path' and 'audio_path' (None if not saved).
    """
    _ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{event_id}_{timestamp}" if event_id is not None else timestamp

    saved_image = None
    saved_audio = None

    # Save face image
    if face_image is not None and face_image.size > 0:
        img_path = os.path.join(PROOF_IMAGES_DIR, f"image_{tag}.jpg")
        try:
            cv2.imwrite(img_path, face_image)
            saved_image = img_path
            print(f"[ProofStorage] Face image saved: {img_path}")
        except Exception as e:
            print(f"[ProofStorage] Failed to save image: {e}")

    # Copy audio file
    if audio_wav_path and os.path.isfile(audio_wav_path):
        audio_path = os.path.join(PROOF_AUDIO_DIR, f"audio_{tag}.wav")
        try:
            shutil.copy2(audio_wav_path, audio_path)
            saved_audio = audio_path
            print(f"[ProofStorage] Audio clip saved: {audio_path}")
        except Exception as e:
            print(f"[ProofStorage] Failed to save audio: {e}")

    return {"image_path": saved_image, "audio_path": saved_audio}
