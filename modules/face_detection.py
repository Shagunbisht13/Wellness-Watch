"""
Standalone Haar Cascade demo (not part of the live pipeline).
Canonical face crop API: ``modules.video_processing.detect_and_crop_face``.
"""

from modules.video_processing import detect_and_crop_face
from modules.device_input import connect_webcam
import cv2

if __name__ == "__main__":
    cap = connect_webcam()
    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        face, rect = detect_and_crop_face(frame)
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Face Detection", frame)
        if face is not None:
            cv2.imshow("Cropped Face", face)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
