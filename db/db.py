import sqlite3

DB_PATH = "database/search.db"


def get_connection():
    return sqlite3.connect(DB_PATH)
