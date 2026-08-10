import sqlite3
from datetime import datetime

DB_PATH = "search.db"


def get_connection(): return sqlite3.connect(DB_PATH)


def create_employee_index():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS employee_index (
            url TEXT PRIMARY KEY,
            name TEXT,
            last_seen TEXT,
            content_hash TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_employee_index():
    conn = get_connection()
    rows = conn.execute("SELECT url,name,last_seen,content_hash FROM employee_index").fetchall()
    conn.close()
    return {row[0]: {"name": row[1], "last_seen": row[2], "content_hash": row[3]} for row in rows}


def update_employee_index(url, name, content_hash):
    conn = get_connection()
    conn.execute("""
        INSERT INTO employee_index(url,name,last_seen,content_hash)
        VALUES(?,?,?,?)
        ON CONFLICT(url) DO UPDATE SET
            name=excluded.name,
            last_seen=excluded.last_seen,
            content_hash=excluded.content_hash
    """, (url, name, datetime.now().isoformat(), content_hash))
    conn.commit()
    conn.close()
