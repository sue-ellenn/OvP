from pathlib import Path
import sqlite3
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_DB = BASE_DIR / "data" / "feedback.db"


def get_feedback_connection():
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(FEEDBACK_DB)


def init_feedback_db():
    with get_feedback_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS employee_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_name TEXT NOT NULL,
                employee_url TEXT,
                query TEXT NOT NULL,
                vote INTEGER NOT NULL DEFAULT 1,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(employee_url, query, session_id)
            )
        """)


def save_employee_feedback(name, url, query, session_id):
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_feedback_connection() as conn:
        conn.execute("""
            INSERT INTO employee_feedback
                (employee_name, employee_url, query, vote, session_id, timestamp)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(employee_url, query, session_id)
            DO UPDATE SET
                vote = 1,
                timestamp = excluded.timestamp
        """, (name, url, query, session_id, timestamp))
