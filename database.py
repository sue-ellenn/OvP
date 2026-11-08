import streamlit as st
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
from pathlib import Path
from io import BytesIO
import requests
from collections import defaultdict
import time

# pd.read_csv("created_data/cleaned_data/repo.csv").to_parquet('created_data/cleaned_data/repo.parquet', compression="snappy")

FILES = {
    "search.db": "https://github.com/sue-ellenn/OvP/releases/download/datav1/search.db",
    "embeddings.npy": "https://github.com/sue-ellenn/OvP/releases/download/datav1/embeddings.npy",
    "meta.npy": "https://github.com/sue-ellenn/OvP/releases/download/datav1/meta.npy"
}
# https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy

# BASE_DIR = Path(".")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "search.db"

PATHS = {
    "search.db": DB_PATH,
    "embeddings.npy": DATA_DIR / "embeddings.npy",
    "meta.npy": DATA_DIR / "meta.npy"
}

def download_if_missing():
    for fname, url in FILES.items():
        path = PATHS[fname]
        path.parent.mkdir(parents=True, exist_ok=True)

        with st.spinner(f"Downloading {fname}..."):
            temp_path = path.with_suffix(path.suffix + ".tmp")

            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()

                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            if fname == "search.db":
                with sqlite3.connect(temp_path) as test:
                    tables = {
                        row[0] for row in test.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }

                required = {"employee", "osiris", "repo", "course_employee", "search"}

                if not required.issubset(tables):
                    temp_path.unlink()
                    raise RuntimeError(f"Downloaded database incomplete: {required - tables}")

            temp_path.replace(path)

            print(f"{fname}: {path.stat().st_size / 1024**2:.2f} MB")
# download_if_missing()

@st.cache_resource(show_spinner=False)
def load_resources():
    download_if_missing()

    # conn = sqlite3.connect("search.db", check_same_thread=False)

    # conn.execute("PRAGMA journal_mode=WAL;")
    # conn.execute("PRAGMA synchronous=NORMAL;")
    # conn.execute("PRAGMA temp_store=MEMORY;")
    # conn.execute("PRAGMA cache_size=-64000;")

    emb = np.load(PATHS["embeddings.npy"], mmap_mode="r")
    meta = np.load(PATHS["meta.npy"], allow_pickle=True)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # conn = sqlite3.connect("search.db")
    #
    # print(conn.execute(
    #     "SELECT COUNT(*) FROM search"
    # ).fetchone())
    #
    # print(conn.execute(
    #     "SELECT COUNT(*) FROM employee"
    # ).fetchone())
    #
    # print(conn.execute(
    #     "SELECT COUNT(*) FROM osiris"
    # ).fetchone())
    #
    # print(conn.execute(
    #     "SELECT COUNT(*) FROM repo"
    # ).fetchone())

    print(emb is not None)
    print(meta is not None)

    # print(conn.execute(
    #     "SELECT name FROM sqlite_master WHERE type='table'"
    # ).fetchall())

    # R = pd.read_parquet("created_data/cleaned_data/repo.parquet")
    # E = pd.read_csv("created_data/cleaned_data/employee.csv")
    # O = pd.read_csv("created_data/cleaned_data/osiris.csv")

    return emb, meta, model  # , E, O, R


def get_connection():
    print("DATABASE:", PATHS["search.db"].resolve())
    print("SIZE:", PATHS["search.db"].stat().st_size)

    with sqlite3.connect(PATHS["search.db"]) as test:
        tables = test.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
        """).fetchall()

        print("TABLES:", tables)

    db_path = PATHS["search.db"]

    print("DATABASE PAD:", db_path.resolve())
    print("BESTAAT:", db_path.exists())
    print("GROOTTE:", db_path.stat().st_size if db_path.exists() else 0)

    with sqlite3.connect(db_path) as test_conn:
        print("AANWEZIGE TABELLEN:", test_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall())


    print("----------")
    print("DB_PATH:", DB_PATH.resolve())
    print("PATHS DB:", PATHS["search.db"].resolve())
    print("BESTAAT:", PATHS["search.db"].exists())
    print("GROOTTE:", PATHS["search.db"].stat().st_size)

    db_path = PATHS["search.db"]

    if not db_path.exists() or db_path.stat().st_size == 0:
        raise FileNotFoundError(f"Database missing or empty: {db_path}")

    conn = sqlite3.connect(
        db_path,
        check_same_thread=False
    )

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    required = {"employee", "osiris", "repo", "course_employee", "search"}
    missing = required - {t[0] for t in tables}

    if missing:
        conn.close()
        raise RuntimeError(f"Missing database tables: {missing}")

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-64000;")

    return conn

# def get_connection():
#     db_path = PATHS["search.db"]
#
#     if not db_path.exists():
#         raise FileNotFoundError(f"Database not found: {db_path}")
#
#     conn = sqlite3.connect(db_path, check_same_thread=False)
#
#     tables = conn.execute("""
#         SELECT name FROM sqlite_master WHERE type='table'
#     """).fetchall()
#
#     table_names = [t[0] for t in tables]
#
#     info = {
#         "Database path": str(db_path.resolve()),
#         "File size (MB)": round(db_path.stat().st_size / 1024**2, 2),
#         "Tables": table_names,
#         "Employee count": conn.execute("SELECT COUNT(*) FROM employee").fetchone()[0] if "employee" in table_names else "Missing",
#     }
#
#     st.write("DATABASE DEBUG:", info)
#
#     required = {"employee", "osiris", "repo", "course_employee", "search"}
#     missing = required - set(table_names)
#
#     if missing:
#         conn.close()
#         st.error(f"Missing tables: {missing}")
#         st.stop()
#
#     conn.execute("PRAGMA journal_mode=WAL;")
#     conn.execute("PRAGMA synchronous=NORMAL;")
#     conn.execute("PRAGMA temp_store=MEMORY;")
#     conn.execute("PRAGMA cache_size=-64000;")
#
#     return conn








# embeddings, meta, model = load_resources()
# conn = get_connection()


