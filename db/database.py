"""
db/database.py
--------------
SQLite database setup for CreditStack AI.
Handles schema creation and provides a connection helper.
No external dependencies -- uses Python's built-in sqlite3.
"""

import sqlite3
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "creditstack.db")


def get_conn() -> sqlite3.Connection:
    """Returns a SQLite connection with row_factory for dict-like rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates all tables if they don't already exist. Safe to call on every startup."""
    conn = get_conn()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS applicant_decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id    TEXT    NOT NULL,
            applicant_index INTEGER NOT NULL,
            probability     REAL    NOT NULL,
            decision        TEXT    NOT NULL,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_runs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id        TEXT NOT NULL,
            risk_summary        TEXT,
            compliance_status   TEXT,
            compliance_notes    TEXT,
            final_letter        TEXT,
            created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id    TEXT NOT NULL,
            user_query      TEXT NOT NULL,
            agent_response  TEXT NOT NULL,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS uploaded_datasets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL,
            row_count   INTEGER NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    print(f"[DB] SQLite initialized at: {DB_PATH}")
