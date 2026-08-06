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

BASE_DIR = Path(".")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PATHS = {
    "search.db": BASE_DIR / "search.db",
    "embeddings.npy": DATA_DIR / "embeddings.npy",
    "meta.npy": DATA_DIR / "meta.npy"
}


def download_if_missing():
    for fname, url in FILES.items():
        path = PATHS[fname]
        if not path.exists():
            with st.spinner(f"Downloading {fname}..."):
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)


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

    conn = sqlite3.connect("search.db")

    print(conn.execute(
        "SELECT COUNT(*) FROM search"
    ).fetchone())

    print(conn.execute(
        "SELECT COUNT(*) FROM employee"
    ).fetchone())

    print(conn.execute(
        "SELECT COUNT(*) FROM osiris"
    ).fetchone())

    print(conn.execute(
        "SELECT COUNT(*) FROM repo"
    ).fetchone())

    print(emb is not None)
    print(meta is not None)

    print(conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall())

    # R = pd.read_parquet("created_data/cleaned_data/repo.parquet")
    # E = pd.read_csv("created_data/cleaned_data/employee.csv")
    # O = pd.read_csv("created_data/cleaned_data/osiris.csv")

    return emb, meta, model  # , E, O, R


def get_connection():

   conn = sqlite3.connect(
        "search.db",
        check_same_thread=False
    )
   conn.execute("PRAGMA journal_mode=WAL;")
   conn.execute("PRAGMA synchronous=NORMAL;")
   conn.execute("PRAGMA temp_store=MEMORY;")
   conn.execute("PRAGMA cache_size=-64000;")
   return conn

# embeddings, meta, model = load_resources()
# conn = get_connection()