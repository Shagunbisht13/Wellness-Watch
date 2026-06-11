"""
Module 1 + 3: Device Input + Video Processing
- Connects to webcam (USB or built-in laptop camera).
- Uses Haar Cascade to detect and crop face regions from each frame.
- Provides a thread-safe VideoStream class for use by the main orchestrator.
"""

import cv2
import threading
import numpy as np


# Load Haar Cascade face detector once at module level
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_camera_index():
    """
    Scan available camera indices (0-3) and return the first working one.
    Prefers external USB cameras (index > 0), falls back to built-in (index 0).
    """
    candidates = list(range(4))
    working = []
    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                working.append(idx)
            cap.release()
    if not working:
        raise RuntimeError("No camera device found. Connect a webcam and retry.")
    # Prefer index > 0 (USB external), but fall back to 0 (built-in)
    for idx in working:
        if idx > 0:
            print(f"[Camera] USB camera detected at index {idx}")
            return idx
    print(f"[Camera] Using built-in/default camera at index {working[0]}")
    return working[0]


def detect_and_crop_face(frame):
    """
    Detect the largest face in a BGR frame and return the cropped face region.
    Returns:
        face_img (np.ndarray | None): Cropped face as BGR image, or None if no face detected.
        face_rect (tuple | None): (x, y, w, h) bounding box of detected face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None

    # Pick the largest face by area
    largest = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = largest
    face_img = frame[y:y + h, x:x + w]
    return face_img, (x, y, w, h)


class VideoStream:
    """
    Thread-safe continuous webcam capture.
    Runs a background thread that reads frames at full FPS.
    The latest frame is always available via `get_frame()`.
    """

    def __init__(self, camera_index=None):
        if camera_index is None:
            camera_index = detect_camera_index()
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Start the background capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[Camera] Video stream started.")

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self):
        """Returns the latest captured frame (or None if not yet available)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.cap.release()
        print("[Camera] Video stream stopped.")


if __name__ == "__main__":
    stream = VideoStream()
    stream.start()
    print("Press ESC to quit.")
    while True:
        frame = stream.get_frame()
        if frame is None:
            continue
        face, rect = detect_and_crop_face(frame)
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Camera — Press ESC to quit", frame)
        if face is not None:
            cv2.imshow("Cropped Face", face)
        if cv2.waitKey(1) == 27:
            break
    stream.stop()
    cv2.destroyAllWindows()