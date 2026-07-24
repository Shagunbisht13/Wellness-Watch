import time

import cv2

from modules.device_input import connect_microphone, connect_webcam, start_capture


def test_device_input():
    print("[Test] Starting webcam and microphone validation test...")

    try:
        cap = connect_webcam()
        stream, audio_queue = connect_microphone()
        stream.start()
        print("[Test] Webcam and microphone connected successfully.")

        started_at = time.time()
        window_name = "Device Input Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while time.time() - started_at < 5.0:
            ret, frame = cap.read()
            if not ret:
                print("[Test] Warning: failed to read frame from webcam.")
                break

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print("[Test] Completed live preview for 5 seconds.")

    except Exception as exc:
        print(f"[Test] Error during device input validation: {exc}")

    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Test] Cleaned up all resources.")


if __name__ == "__main__":
    test_device_input()
