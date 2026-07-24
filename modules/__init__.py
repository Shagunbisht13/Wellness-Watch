"""
Harassment Detection System — 13-Module Architecture
====================================================
Module  1  device_input          — USB webcam + mic capture
Module  2  realtime_stream       — threaded live frame/audio queues
Module  3  video_processing      — Haar Cascade face crop
Module  4  audio_processing      — waveform clean-up
Module  5  preprocessing         — image / audio / BERT-ready text
Module  6  speech_to_text        — audio chunk → text
Module  7  feature_extraction    — BERT / CNN / MFCC vectors
Module  8  pretrained_models      — face, speech, text harassment models
Module  9  multimodal_decision   — fuse modalities → verdict
Module 10  proof_storage         — evidence on harassment
Module 11  database_module       — prediction history
Module 12  alert                 — real-time warnings
Module 13  dashboard             — Streamlit / Flask visualization
"""
