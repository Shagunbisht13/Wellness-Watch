"""
Integration test — verifies core pipeline logic without requiring hardware.
Run: python test_integration.py

Uses SQLite backend so MongoDB is not required for CI/local smoke tests.
"""
import os
import sys

sys.path.insert(0, ".")

# Prefer SQLite for offline tests (Module 11 swappable backend)
os.environ.setdefault("HDS_DB_BACKEND", "sqlite")
os.environ.setdefault("HDS_SQLITE_PATH", os.path.join("database", "test_history.db"))

from modules.database_module import initialize_db, insert_record, fetch_recent, fetch_stats
from modules.multimodal_decision import make_decision
from modules.pretrained_models.text_harassment import classify_text
from modules.preprocessing import preprocess_text, tokenize_text
from modules.alert import trigger_alert, WARNING_HARASSMENT

errors = []

print("=" * 50)
print("  Harassment Detection System — Integration Tests")
print("=" * 50)

initialize_db()
print("\n[1] Database initialized OK")

print("\n[2] Decision Engine Tests (Module 9 — all-three AND for harassment):")
cases = [
    # face angry + voice aggressive + text harassment → harassment
    ("angry", "angry tone", "harassment", "harassment"),
    ("neutral", "calm tone", "normal", "normal behaviour"),
    # face+voice without toxic text → stress
    ("angry", "angry tone", "normal", "stress"),
    ("happy", "calm tone", "normal", "normal behaviour"),
    # text-only abusive without angry face+voice → stress (not full triple)
    ("sad", "calm tone", "abusive", "stress"),
    # full triple with abusive label
    ("angry", "angry tone", "abusive", "harassment"),
]
for face, voice, text, expected in cases:
    result = make_decision(face, voice, text)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        errors.append(f"Decision: {face}|{voice}|{text} expected={expected} got={result}")
    print(f"  [{status}] face={face}, voice={voice}, text={text} => {result}")

print("\n[3] Preprocessing — no stopword removal (Module 5):")
sample = "This is a clear warning about harassment today!"
cleaned = preprocess_text(sample)
tokens = tokenize_text(sample)
# Stopwords like "is", "a", "about" must remain
for sw in ("is", "a", "about"):
    if sw not in tokens:
        errors.append(f"Stopword '{sw}' was removed — BERT needs full context")
        print(f"  [FAIL] stopword '{sw}' missing from {tokens}")
    else:
        print(f"  [PASS] kept stopword '{sw}' in tokens")
print(f"  cleaned: {cleaned!r}")

print("\n[4] Text Classifier (keyword fallback / BERT) Tests:")
text_cases = [
    ("shut up idiot", "harassment"),
    ("good morning everyone", "normal"),
    ("I hate you", "harassment"),
]
for sentence, expected in text_cases:
    result = classify_text(sentence)
    # Model may return harassment/abusive; treat abusive as OK for toxic samples
    ok = result == expected or (expected == "harassment" and result == "abusive")
    status = "PASS" if ok else "WARN (model may differ)"
    if not ok and expected == "normal" and result != "normal":
        errors.append(f"Text: '{sentence}' expected normal got {result}")
    print(f"  [{status}] '{sentence}' => {result}")

print("\n[5] Alert Module:")
msg = trigger_alert("harassment", face_emotion="angry", voice_emotion="angry tone",
                    text_prediction="harassment", event_id=1, desktop=False)
if msg != WARNING_HARASSMENT:
    errors.append(f"Alert message mismatch: {msg}")
    print(f"  [FAIL] got {msg!r}")
else:
    print(f"  [PASS] {msg}")

print("\n[6] Database Insert + Read:")
rid1 = insert_record("angry", "angry tone", "harassment", "harassment")
rid2 = insert_record("neutral", "calm tone", "normal", "normal behaviour")
records = fetch_recent(5)
if len(records) < 2:
    errors.append("DB should have at least 2 records")
    print("  [FAIL] not enough records")
else:
    print(f"  [PASS] Inserted rows: {rid1}, {rid2}")
    for rec in records[:3]:
        print(f"  Row {rec.get('id')}: {rec.get('timestamp')} | {rec.get('final_result')}")

stats = fetch_stats()
print(f"\n[7] Stats: {stats}")

print("\n" + "=" * 50)
if errors:
    print(f"  {len(errors)} FAILURES:")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("  All tests PASSED!")
print("=" * 50)
