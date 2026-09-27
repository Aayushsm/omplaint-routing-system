"""
database.py
------------
Handles all persistence for the complaint routing system using SQLite
(built into Python -- no external database server required).

Table: complaints
    id                 INTEGER PRIMARY KEY AUTOINCREMENT
    complaint_id        TEXT    (human-friendly id, e.g. CMP-0001)
    complaint_text       TEXT
    predicted_category   TEXT
    confidence           REAL    (0.0 - 1.0)
    department            TEXT
    status                TEXT
    timestamp             TEXT    (ISO format string)
"""

import os
import sqlite3
from datetime import datetime
from contextlib import contextmanager

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "complaints.db")


@contextmanager
def get_connection():
    """Context manager that yields a SQLite connection and always
    closes it, even if an error occurs -- avoids leaking file handles."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create the complaints table if it doesn't already exist. Safe to
    call every time the app starts."""
    try:
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id TEXT UNIQUE NOT NULL,
                    complaint_text TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    department TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to initialise the database: {e}")


def _generate_next_complaint_id(conn) -> str:
    """Generate the next human-friendly complaint id, e.g. CMP-0001."""
    cursor = conn.execute("SELECT COUNT(*) as cnt FROM complaints")
    count = cursor.fetchone()["cnt"]
    return f"CMP-{count + 1:04d}"


def insert_complaint(complaint_text, predicted_category, confidence, department,
                      status="Automatically Routed"):
    """
    Insert a new complaint record into the database.

    Returns
    -------
    dict
        The full complaint record that was inserted (including the
        generated complaint_id and timestamp), useful for immediately
        displaying a result card in the UI.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_connection() as conn:
            complaint_id = _generate_next_complaint_id(conn)
            conn.execute(
                """
                INSERT INTO complaints
                    (complaint_id, complaint_text, predicted_category, confidence,
                     department, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (complaint_id, complaint_text, predicted_category, confidence,
                 department, status, timestamp),
            )
            conn.commit()
        return {
            "complaint_id": complaint_id,
            "complaint_text": complaint_text,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "department": department,
            "status": status,
            "timestamp": timestamp,
        }
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to save complaint to the database: {e}")


def fetch_all_complaints():
    """Return all complaints as a list of dicts, most recent first."""
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM complaints ORDER BY id DESC"
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to fetch complaints from the database: {e}")


def update_status(complaint_id: str, new_status: str):
    """Update the status of a specific complaint (used by the dashboard)."""
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE complaints SET status = ? WHERE complaint_id = ?",
                (new_status, complaint_id),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to update complaint status: {e}")


def get_complaint_count():
    """Return the total number of complaints stored so far."""
    try:
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM complaints")
            return cursor.fetchone()["cnt"]
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to count complaints: {e}")
