# src/feedback_store.py

"""
User feedback store.

Persists thumbs up/down ratings per (user_id, dish_id) using SQLite.
Design: append-only writes. Every rating is a new row with a timestamp —
we never UPDATE or DELETE. This makes it trivial to replay history,
compute decay-weighted scores later (Week 3 hybrid merge), and debug
"why did the system recommend X" after the fact.

Usage:
    from src.feedback_store import FeedbackStore

    store = FeedbackStore()
    store.add_feedback(user_id="u1", dish_name="Dal Makhani", rating=1)   # thumbs up
    store.add_feedback(user_id="u1", dish_name="Rasgulla", rating=-1)    # thumbs down

    history = store.get_user_history("u1")
    matrix = store.get_user_item_matrix()
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import pandas as pd

DB_PATH = Path("data/feedback.db")


class FeedbackStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    dish_name TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                    timestamp TEXT NOT NULL
                )
            """)
            # Index speeds up the two queries we run constantly:
            # "all feedback for this user" and "all feedback for this dish"
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON feedback(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dish ON feedback(dish_name)")
            conn.commit()

    def add_feedback(self, user_id: str, dish_name: str, rating: int) -> dict:
        """
        rating must be 1 (thumbs up) or -1 (thumbs down).
        Always inserts a new row — never overwrites previous feedback for
        the same user+dish, since a user's opinion can change over time
        and we want the full history for decay-weighting later.
        """
        if rating not in (1, -1):
            raise ValueError("rating must be 1 (thumbs up) or -1 (thumbs down)")

        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO feedback (user_id, dish_name, rating, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, dish_name, rating, timestamp),
            )
            conn.commit()
            return {
                "id": cursor.lastrowid,
                "user_id": user_id,
                "dish_name": dish_name,
                "rating": rating,
                "timestamp": timestamp,
            }

    def get_user_history(self, user_id: str) -> list[dict]:
        """All feedback rows for one user, oldest first."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT user_id, dish_name, rating, timestamp FROM feedback "
                "WHERE user_id = ? ORDER BY timestamp ASC",
                (user_id,),
            )
            rows = cursor.fetchall()
        return [
            {"user_id": r[0], "dish_name": r[1], "rating": r[2], "timestamp": r[3]}
            for r in rows
        ]

    def get_liked_dishes(self, user_id: str) -> list[str]:
        """Dish names this user gave a thumbs-up to (most recent rating wins per dish)."""
        latest = self._latest_rating_per_dish(user_id)
        return [dish for dish, rating in latest.items() if rating == 1]

    def get_disliked_dishes(self, user_id: str) -> list[str]:
        """Dish names this user gave a thumbs-down to (most recent rating wins per dish)."""
        latest = self._latest_rating_per_dish(user_id)
        return [dish for dish, rating in latest.items() if rating == -1]

    def _latest_rating_per_dish(self, user_id: str) -> dict:
        """Internal: collapse history to the most recent rating per dish."""
        history = self.get_user_history(user_id)  # oldest -> newest
        latest = {}
        for row in history:
            latest[row["dish_name"]] = row["rating"]  # later rows overwrite earlier
        return latest

    def feedback_count(self, user_id: str) -> int:
        """Total number of feedback events for this user (used to gate CF vs baseline)."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE user_id = ?", (user_id,)
            )
            return cursor.fetchone()[0]

    def get_user_item_matrix(self) -> pd.DataFrame:
        """
        Returns a user x dish matrix of ratings (most recent rating per pair),
        suitable for collaborative filtering. Missing entries are 0 (no signal),
        not to be confused with an actual neutral rating.
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT user_id, dish_name, rating, timestamp FROM feedback", conn
            )
        if df.empty:
            return pd.DataFrame()

        # Keep only the most recent rating per (user_id, dish_name) pair
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["user_id", "dish_name"], keep="last"
        )
        matrix = df.pivot(index="user_id", columns="dish_name", values="rating").fillna(0)
        return matrix

    def all_feedback_df(self) -> pd.DataFrame:
        """Raw feedback table as a DataFrame — useful for debugging/eval."""
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM feedback", conn)


# ── Quick manual test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    store = FeedbackStore(db_path=Path("data/feedback_demo.db"))

    store.add_feedback("user_1", "Dal Makhani", 1)
    store.add_feedback("user_1", "Rasgulla", -1)
    store.add_feedback("user_1", "Shahi Paneer", 1)
    store.add_feedback("user_2", "Dal Makhani", 1)
    store.add_feedback("user_2", "Mutton Rogan Josh", 1)

    print("User 1 history:", store.get_user_history("user_1"))
    print("User 1 liked:", store.get_liked_dishes("user_1"))
    print("User 1 disliked:", store.get_disliked_dishes("user_1"))
    print("User 1 feedback count:", store.feedback_count("user_1"))
    print()
    print("User-item matrix:")
    print(store.get_user_item_matrix())
