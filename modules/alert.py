"""
Module 12 — Alert Module
========================
Purpose:
    Notify in real time when harassment is detected.

Technologies:
    Python notifications (console + optional desktop toast via plyer).
    Dashboard/Streamlit surfaces consume the same message string.

Example:
    "WARNING: Harassment detected"

Output:
    Alert message string; side-effect notifications when enabled.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Deque, Optional

_ALERT_LOG: Deque[dict] = deque(maxlen=200)
_lock = threading.Lock()

WARNING_HARASSMENT = "WARNING: Harassment detected"
WARNING_STRESS = "WARNING: Elevated stress indicators"


def _try_desktop_notify(title: str, message: str) -> None:
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=5)
    except Exception:
        # Desktop notifications are optional
        pass


def trigger_alert(
    final_result: str,
    face_emotion: str = "—",
    voice_emotion: str = "—",
    text_prediction: str = "—",
    event_id: Optional[int] = None,
    desktop: bool = True,
) -> Optional[str]:
    """
    Fire a real-time alert when the decision is harassment (or stress).

    Returns the alert message, or None if no alert is warranted.
    """
    result = (final_result or "").lower().strip()
    if result == "harassment":
        message = WARNING_HARASSMENT
    elif result == "stress":
        message = WARNING_STRESS
    else:
        return None

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "final_result": final_result,
        "face_emotion": face_emotion,
        "speech_emotion": voice_emotion,
        "text_prediction": text_prediction,
        "event_id": event_id,
    }
    with _lock:
        _ALERT_LOG.appendleft(entry)

    banner = f"\n{'!' * 60}\n  {message}"
    if event_id is not None:
        banner += f" — Event #{event_id}"
    banner += f"\n{'!' * 60}\n"
    print(banner)

    if desktop and result == "harassment":
        _try_desktop_notify("Harassment Detection System", message)

    return message


def get_alert_log(limit: int = 50) -> list[dict]:
    """Return recent alert entries (newest first)."""
    with _lock:
        return list(_ALERT_LOG)[:limit]


def clear_alert_log() -> None:
    with _lock:
        _ALERT_LOG.clear()
