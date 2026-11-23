"""SQLite database utility for logging user feedback in CompressRoute Streamlit app."""

import sqlite3
import time
from typing import Dict, Any, List


DB_PATH = "feedback.db"


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes the feedback table in SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            recommended_model TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comments TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(
    query: str,
    recommended_model: str,
    rating: int,
    comments: str = "",
    db_path: str = DB_PATH
) -> int:
    """Inserts a user feedback record (rating: +1 for thumbs up, -1 for thumbs down)."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO feedback (timestamp, query, recommended_model, rating, comments)
        VALUES (?, ?, ?, ?, ?)
    """, (now, query, recommended_model, rating, comments))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_feedback_records(limit: int = 50, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent feedback records."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, query, recommended_model, rating, comments
        FROM feedback
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "query": r[2],
            "recommended_model": r[3],
            "rating": "👍 Positive" if r[4] == 1 else "👎 Negative",
            "comments": r[5] or ""
        }
        for r in rows
    ]
