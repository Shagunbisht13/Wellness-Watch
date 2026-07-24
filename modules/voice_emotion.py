""" Backward-compatible shim — use ``modules.pretrained_models.speech_emotion``. """
from modules.pretrained_models.speech_emotion import (  # noqa: F401
    predict_speech_emotion,
    predict_voice_emotion,
)
