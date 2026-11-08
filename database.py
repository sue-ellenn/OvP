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
    "search.db": "https://github.com/sue-ellenn/OvP/releases/download/data/search.db",
    "embeddings.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy",
    "meta.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/meta.npy"
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
            r = requests.get(url, timeout=180)
            r.raise_for_status()

            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_bytes(r.content)

            if temp_path.stat().st_size == 0:
                temp_path.unlink()
                raise RuntimeError(f"Empty download: {fname}")

            temp_path.replace(path)

        print(f"Downloaded {fname}: {path.stat().st_size} bytes")

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
    db_path = PATHS["search.db"]

    print("DATABASE PAD:", db_path.resolve())
    print("BESTAAT:", db_path.exists())
    print("GROOTTE:", db_path.stat().st_size if db_path.exists() else 0)

    with sqlite3.connect(db_path) as test_conn:
        print("AANWEZIGE TABELLEN:", test_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall())

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
# embeddings, meta, model = load_resources()
# conn = get_connection()
