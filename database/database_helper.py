"""
Module 11: Database Helper (MongoDB)
- Manages the MongoDB connection to store prediction history.
- Uses pymongo to connect to localhost:27017.
- Provides functions to: initialize the db, insert a record, query recent logs.

Collection schema (history):
    _id             ObjectId
    timestamp       TEXT    — ISO format datetime
    face_emotion    TEXT
    speech_emotion  TEXT
    text_prediction TEXT
    final_result    TEXT
    proof_image     TEXT    — path to saved image (or None)
    proof_audio     TEXT    — path to saved audio (or None)
"""

import os
from datetime import datetime

# Import pymongo
try:
    from pymongo import MongoClient
except ImportError:
    print("[DB] pymongo not installed. Run: pip install pymongo")
    MongoClient = None

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "harassment_detection"
COLLECTION_NAME = "history"

_client = None
_db = None
_collection = None

def _get_collection():
    global _client, _db, _collection
    if _collection is None and MongoClient is not None:
        try:
            _client = MongoClient(MONGO_URI)
            _db = _client[DB_NAME]
            _collection = _db[COLLECTION_NAME]
        except Exception as e:
            print(f"[DB] MongoDB Connection Error: {e}")
    return _collection


def initialize_db():
    """Verify MongoDB connection (Collections are created automatically on insert)."""
    coll = _get_collection()
    if coll is not None:
        try:
            # Ping the server to verify connection
            _client.admin.command('ping')
            print(f"[DB] Connected to MongoDB at {MONGO_URI}")
        except Exception as e:
            print(f"[DB] Failed to connect to MongoDB: {e}")
    else:
        print("[DB] Warning: MongoDB client not available.")


def insert_record(
    face_emotion: str,
    speech_emotion: str,
    text_prediction: str,
    final_result: str,
    proof_image: str | None = None,
    proof_audio: str | None = None,
) -> str:
    """
    Insert a prediction record into the history collection.

    Returns:
        The stringified ObjectId of the newly inserted record.
    """
    coll = _get_collection()
    if coll is None:
        return "error_no_db"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    document = {
        "timestamp": timestamp,
        "face_emotion": face_emotion,
        "speech_emotion": speech_emotion,
        "text_prediction": text_prediction,
        "final_result": final_result,
        "proof_image": proof_image,
        "proof_audio": proof_audio
    }
    
    try:
        result = coll.insert_one(document)
        inserted_id = str(result.inserted_id)
        print(f"[DB] Record {inserted_id} saved: {final_result} at {timestamp}")
        return inserted_id
    except Exception as e:
        print(f"[DB] Failed to insert record: {e}")
        return "error_insert"


def fetch_recent(limit: int = 50) -> list[dict]:
    """
    Fetch the most recent `limit` records from the history collection.

    Returns:
        List of dicts, newest first. _id is converted to string for JSON serialization.
    """
    coll = _get_collection()
    if coll is None:
        return []

    try:
        # Sort by _id descending (which sorts by creation time)
        cursor = coll.find().sort("_id", -1).limit(limit)
        results = []
        for doc in cursor:
            # Convert ObjectId to string for easy JSON serialization in Flask
            doc["_id"] = str(doc["_id"])
            # Ensure compatibility with existing code expecting 'id'
            doc["id"] = doc["_id"]
            results.append(doc)
        return results
    except Exception as e:
        print(f"[DB] Error fetching recent records: {e}")
        return []


def fetch_stats() -> dict:
    """
    Return aggregate counts per final_result for dashboard statistics.
    """
    coll = _get_collection()
    if coll is None:
        return {}

    try:
        pipeline = [
            {"$group": {"_id": "$final_result", "count": {"$sum": 1}}}
        ]
        results = coll.aggregate(pipeline)
        
        stats = {}
        for row in results:
            if row["_id"]: # Exclude None if any
                stats[row["_id"]] = row["count"]
        return stats
    except Exception as e:
        print(f"[DB] Error fetching stats: {e}")
        return {}


if __name__ == "__main__":
    initialize_db()
    
    # Test inserts
    id1 = insert_record("angry", "angry tone", "harassment", "harassment")
    id2 = insert_record("neutral", "calm tone", "normal", "normal behaviour")
    
    print("\nRecent 5 records:")
    recent = fetch_recent(5)
    for r in recent:
        print(r)
        
    print("\nStats:")
    print(fetch_stats())
