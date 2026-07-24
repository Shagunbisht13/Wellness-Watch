"""
Module 11 — Database Module
===========================
Purpose:
    Store full prediction history for dashboard and audits.

Technologies:
    Swappable backends — MongoDB (default), SQLite, or CSV.
    Set env ``HDS_DB_BACKEND`` to ``mongo`` | ``sqlite`` | ``csv``.

Fields:
    timestamp, face_emotion, speech_emotion, text_prediction, final_result
    (+ optional proof_image, proof_audio)

Example row:
    10:32 | angry | abusive | harassment

Output:
    Inserted record id; query helpers for recent rows and aggregate stats.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

# Default: mongo (existing deployment). Fallbacks: sqlite, csv.
BACKEND = os.environ.get("HDS_DB_BACKEND", "mongo").lower()
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "harassment_detection"
COLLECTION_NAME = "history"
SQLITE_PATH = os.environ.get("HDS_SQLITE_PATH", os.path.join("database", "history.db"))
CSV_PATH = os.environ.get("HDS_CSV_PATH", os.path.join("database", "history.csv"))

_FIELDS = (
    "timestamp",
    "face_emotion",
    "speech_emotion",
    "text_prediction",
    "final_result",
    "proof_image",
    "proof_audio",
)


class HistoryBackend(ABC):
    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def insert(
        self,
        face_emotion: str,
        speech_emotion: str,
        text_prediction: str,
        final_result: str,
        proof_image: Optional[str] = None,
        proof_audio: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    def fetch_recent(self, limit: int = 50) -> list[dict]: ...

    @abstractmethod
    def fetch_stats(self) -> dict: ...


class MongoBackend(HistoryBackend):
    def __init__(self):
        self._client = None
        self._collection = None
        try:
            from pymongo import MongoClient
            self._MongoClient = MongoClient
        except ImportError:
            print("[DB] pymongo not installed. Run: pip install pymongo")
            self._MongoClient = None

    def _coll(self):
        if self._collection is None and self._MongoClient is not None:
            try:
                self._client = self._MongoClient(MONGO_URI)
                self._collection = self._client[DB_NAME][COLLECTION_NAME]
            except Exception as e:
                print(f"[DB] MongoDB Connection Error: {e}")
        return self._collection

    def initialize(self) -> None:
        coll = self._coll()
        if coll is not None:
            try:
                self._client.admin.command("ping")
                print(f"[DB] Connected to MongoDB at {MONGO_URI}")
            except Exception as e:
                print(f"[DB] Failed to connect to MongoDB: {e}")
        else:
            print("[DB] Warning: MongoDB client not available.")

    def insert(
        self,
        face_emotion: str,
        speech_emotion: str,
        text_prediction: str,
        final_result: str,
        proof_image: Optional[str] = None,
        proof_audio: Optional[str] = None,
    ) -> str:
        coll = self._coll()
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
            "proof_audio": proof_audio,
        }
        try:
            result = coll.insert_one(document)
            inserted_id = str(result.inserted_id)
            print(f"[DB] Record {inserted_id} saved: {final_result} at {timestamp}")
            return inserted_id
        except Exception as e:
            print(f"[DB] Failed to insert record: {e}")
            return "error_insert"

    def fetch_recent(self, limit: int = 50) -> list[dict]:
        coll = self._coll()
        if coll is None:
            return []
        try:
            results = []
            for doc in coll.find().sort("_id", -1).limit(limit):
                doc["_id"] = str(doc["_id"])
                doc["id"] = doc["_id"]
                results.append(doc)
            return results
        except Exception as e:
            print(f"[DB] Error fetching recent records: {e}")
            return []

    def fetch_stats(self) -> dict:
        coll = self._coll()
        if coll is None:
            return {}
        try:
            pipeline = [{"$group": {"_id": "$final_result", "count": {"$sum": 1}}}]
            return {
                row["_id"]: row["count"]
                for row in coll.aggregate(pipeline)
                if row["_id"]
            }
        except Exception as e:
            print(f"[DB] Error fetching stats: {e}")
            return {}


class SQLiteBackend(HistoryBackend):
    def __init__(self, path: str = SQLITE_PATH):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    face_emotion TEXT,
                    speech_emotion TEXT,
                    text_prediction TEXT,
                    final_result TEXT,
                    proof_image TEXT,
                    proof_audio TEXT
                )
                """
            )
            conn.commit()
        print(f"[DB] SQLite ready at {self.path}")

    def insert(
        self,
        face_emotion: str,
        speech_emotion: str,
        text_prediction: str,
        final_result: str,
        proof_image: Optional[str] = None,
        proof_audio: Optional[str] = None,
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO history
                (timestamp, face_emotion, speech_emotion, text_prediction,
                 final_result, proof_image, proof_audio)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    face_emotion,
                    speech_emotion,
                    text_prediction,
                    final_result,
                    proof_image,
                    proof_audio,
                ),
            )
            conn.commit()
            rid = str(cur.lastrowid)
        print(f"[DB] Record {rid} saved: {final_result} at {timestamp}")
        return rid

    def fetch_recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_stats(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT final_result, COUNT(*) AS count FROM history "
                "GROUP BY final_result"
            ).fetchall()
        return {r["final_result"]: r["count"] for r in rows if r["final_result"]}


class CSVBackend(HistoryBackend):
    def __init__(self, path: str = CSV_PATH):
        self.path = path

    def initialize(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        if not os.path.isfile(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=("id",) + _FIELDS)
                writer.writeheader()
        print(f"[DB] CSV ready at {self.path}")

    def insert(
        self,
        face_emotion: str,
        speech_emotion: str,
        text_prediction: str,
        final_result: str,
        proof_image: Optional[str] = None,
        proof_audio: Optional[str] = None,
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = self.fetch_recent(limit=10_000)
        next_id = str(len(rows) + 1)
        row = {
            "id": next_id,
            "timestamp": timestamp,
            "face_emotion": face_emotion,
            "speech_emotion": speech_emotion,
            "text_prediction": text_prediction,
            "final_result": final_result,
            "proof_image": proof_image or "",
            "proof_audio": proof_audio or "",
        }
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("id",) + _FIELDS)
            writer.writerow(row)
        print(f"[DB] Record {next_id} saved: {final_result} at {timestamp}")
        return next_id

    def fetch_recent(self, limit: int = 50) -> list[dict]:
        if not os.path.isfile(self.path):
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows.reverse()
        return rows[:limit]

    def fetch_stats(self) -> dict:
        stats: dict[str, int] = {}
        for row in self.fetch_recent(limit=100_000):
            key = row.get("final_result") or ""
            if key:
                stats[key] = stats.get(key, 0) + 1
        return stats


def _make_backend() -> HistoryBackend:
    if BACKEND == "sqlite":
        return SQLiteBackend()
    if BACKEND == "csv":
        return CSVBackend()
    return MongoBackend()


_backend: HistoryBackend = _make_backend()


def initialize_db() -> None:
    """Initialize the active history backend."""
    _backend.initialize()


def insert_record(
    face_emotion: str,
    speech_emotion: str,
    text_prediction: str,
    final_result: str,
    proof_image: str | None = None,
    proof_audio: str | None = None,
) -> str:
    """Insert a prediction record. Returns the new record id."""
    return _backend.insert(
        face_emotion=face_emotion,
        speech_emotion=speech_emotion,
        text_prediction=text_prediction,
        final_result=final_result,
        proof_image=proof_image,
        proof_audio=proof_audio,
    )


def fetch_recent(limit: int = 50) -> list[dict]:
    """Fetch the most recent `limit` records (newest first)."""
    return _backend.fetch_recent(limit=limit)


def fetch_stats() -> dict:
    """Return aggregate counts per final_result."""
    return _backend.fetch_stats()


if __name__ == "__main__":
    initialize_db()
    insert_record("angry", "angry tone", "harassment", "harassment")
    insert_record("neutral", "calm tone", "normal", "normal behaviour")
    print(fetch_recent(5))
    print(fetch_stats())
