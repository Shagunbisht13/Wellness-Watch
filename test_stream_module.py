import time

from modules.realtime_stream import StreamManager


def test_stream_module():
    manager = StreamManager(frame_fps=18, audio_chunk_duration=4.0)
    manager.start()

    frame_count = 0
    chunk_count = 0
    start_time = time.time()
    run_duration = 10.0

    try:
        while time.time() - start_time < run_duration:
            frame = manager.get_frame()
            if frame is not None:
                frame_count += 1
                print(f"[Test] Received frame #{frame_count}")

            chunk = manager.get_audio_chunk()
            if chunk is not None:
                chunk_count += 1
                print(
                    f"[Test] Received audio chunk #{chunk_count} "
                    f"({len(chunk)} samples)"
                )

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[Test] Interrupted by user.")
    finally:
        manager.stop()
        print(
            f"[Test] Completed. Frames received: {frame_count}, "
            f"audio chunks received: {chunk_count}"
        )


if __name__ == "__main__":
    test_stream_module()
