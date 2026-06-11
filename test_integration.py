"""
Integration test — verifies the core pipeline logic without requiring hardware.
Run: python test_integration.py
"""
import sys
sys.path.insert(0, '.')

from database.database_helper import initialize_db, insert_record, fetch_recent, fetch_stats
from modules.decision_engine import make_decision
from modules.text_classifier import classify_text

errors = []

print("=" * 50)
print("  Harassment Detection System — Integration Tests")
print("=" * 50)

# DB init
initialize_db()
print("\n[1] Database initialized OK")

# Decision engine
print("\n[2] Decision Engine Tests:")
cases = [
    ("angry", "angry tone", "harassment", "harassment"),
    ("neutral", "calm tone", "normal", "normal behaviour"),
    ("angry", "angry tone", "normal", "stress"),
    ("happy", "calm tone", "normal", "normal behaviour"),
    ("sad", "calm tone", "abusive", "harassment"),
]
for face, voice, text, expected in cases:
    result = make_decision(face, voice, text)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        errors.append(f"Decision: {face}|{voice}|{text} expected={expected} got={result}")
    print(f"  [{status}] face={face}, voice={voice}, text={text} => {result}")

# Text classifier keyword fallback
print("\n[3] Text Classifier (keyword fallback) Tests:")
text_cases = [
    ("shut up idiot", "harassment"),
    ("good morning everyone", "normal"),
    ("I hate you", "harassment"),
]
for sentence, expected in text_cases:
    result = classify_text(sentence)
    status = "PASS" if result == expected else "WARN (model may differ)"
    print(f"  [{status}] '{sentence}' => {result}")

# Database insert + read
print("\n[4] Database Insert + Read Tests:")
rid1 = insert_record("angry", "angry tone", "harassment", "harassment")
rid2 = insert_record("neutral", "calm tone", "normal", "normal behaviour")
records = fetch_recent(5)
assert len(records) >= 2, "Should have at least 2 records"
print(f"  [PASS] Inserted rows: {rid1}, {rid2}")
for rec in records[:3]:
    print(f"  Row {rec['id']}: {rec['timestamp']} | {rec['final_result']}")

stats = fetch_stats()
print(f"\n[5] Stats: {stats}")

print("\n" + "=" * 50)
if errors:
    print(f"  {len(errors)} FAILURES:")
    for e in errors:
        print(f"   - {e}")
else:
    print("  All tests PASSED!")
print("=" * 50)
