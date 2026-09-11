# -*- coding: utf-8 -*-
"""数据存储层:样本、路由日志、模型元信息,均落在 SQLite 里"""

import hashlib
import sqlite3
import threading
import time
from typing import Optional

import numpy as np


class SampleStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT UNIQUE,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    label INTEGER NOT NULL,
                    raw_score INTEGER,
                    source TEXT NOT NULL,
                    was_uncertain INTEGER NOT NULL DEFAULT 0,
                    used_in_training INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_meta (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL,
                    trained_at REAL NOT NULL,
                    n_samples INTEGER NOT NULL,
                    ambiguous_threshold_used REAL NOT NULL,
                    val_accuracy REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    was_uncertain INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add_sample(self, text: str, embedding: np.ndarray, label: int,
                    source: str, raw_score: Optional[int] = None,
                    was_uncertain: bool = False):
        text_hash = self._hash_text(text)
        emb_blob = embedding.astype(np.float32).tobytes()
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    """INSERT INTO samples
                       (text_hash, text, embedding, label, raw_score, source,
                        was_uncertain, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (text_hash, text, emb_blob, label, raw_score, source,
                     int(was_uncertain), time.time())
                )
            except sqlite3.IntegrityError:
                pass  # 重复文本,跳过

    def log_routing(self, was_uncertain: bool):
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO routing_log (was_uncertain, created_at) VALUES (?, ?)",
                (int(was_uncertain), time.time())
            )

    def count_new_samples_since_last_train(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE used_in_training = 0"
            ).fetchone()
            return row[0]

    def get_recent_ambiguous_rate(self, window: int) -> float:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT was_uncertain FROM routing_log ORDER BY id DESC LIMIT ?",
                (window,)
            ).fetchall()
        if not rows:
            return 0.0
        uncertain_count = sum(r[0] for r in rows)
        return uncertain_count / len(rows)

    def get_training_data(self, max_samples: Optional[int] = None):
        query = "SELECT embedding, label FROM samples ORDER BY id DESC"
        if max_samples:
            query += f" LIMIT {int(max_samples)}"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        if not rows:
            return None, None
        X = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
        y = np.array([r[1] for r in rows])
        return X, y

    def mark_all_used_in_training(self):
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE samples SET used_in_training = 1")

    def save_model_meta(self, version: int, n_samples: int,
                         ambiguous_threshold_used: float, val_accuracy: Optional[float]):
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO model_meta
                   (version, trained_at, n_samples, ambiguous_threshold_used, val_accuracy)
                   VALUES (?, ?, ?, ?, ?)""",
                (version, time.time(), n_samples, ambiguous_threshold_used, val_accuracy)
            )

    def get_latest_model_version(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(version) FROM model_meta").fetchone()
            return row[0] or 0

    def get_retrain_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM model_meta").fetchone()
            return row[0]
