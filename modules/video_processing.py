"""
Module 3 — Video Processing Module
==================================
Purpose:
    Extract the face region from a webcam frame so downstream emotion
    models see only the face (background removed).

Technologies:
    OpenCV + Haar Cascade (haarcascade_frontalface_default.xml).
    MediaPipe Face Detection may be swapped in later if needed.

Steps:
    capture frame → detect face → crop face region → send to preprocessing

Output:
    Cropped face image (BGR numpy array) and optional bounding box.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

# Haar Cascade face detector (loaded once)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_crop_face(
    frame: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect the largest face in a BGR frame and return the cropped region.

    Returns
    -------
    face_img : np.ndarray | None
        Cropped face as BGR image, or None if no face detected.
    face_rect : (x, y, w, h) | None
        Bounding box of the detected face.
    """
    if frame is None or frame.size == 0:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None, None

    largest = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = largest
    face_img = frame[y : y + h, x : x + w]
    return face_img, (int(x), int(y), int(w), int(h))


if __name__ == "__main__":
    from modules.device_input import connect_webcam

    cap = connect_webcam()
    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        face, rect = detect_and_crop_face(frame)
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Video Processing — Haar Cascade", frame)
        if face is not None:
            cv2.imshow("Cropped Face", face)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
