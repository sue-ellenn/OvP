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

# pd.read_csv("created_data/cleaned_data/repo.csv").to_parquet('created_data/cleaned_data/repo.parquet', compression="snappy")

FILES = {
    "search.db": "https://github.com/sue-ellenn/OvP/releases/download/data/search.db",
    "embeddings.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy",
    "meta.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/meta.npy"
}
# https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = "search.db"
EMBEDDINGS_PATH = FILES["embeddings.npy"]

response = requests.get(EMBEDDINGS_PATH)
response.raise_for_status()

embeddings = np.load(BytesIO(response.content))
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def download_if_missing():
    for fname, url in FILES.items():
        path = DATA_DIR / fname
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
    conn = sqlite3.connect(DATA_DIR / "search.db", check_same_thread=False)
    emb = np.load(DATA_DIR / "embeddings.npy", mmap_mode="r")
    meta = np.load(DATA_DIR / "meta.npy", allow_pickle=True)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    R = pd.read_parquet("created_data/cleaned_data/repo.parquet")
    E = pd.read_csv("created_data/cleaned_data/employee.csv")
    O = pd.read_csv("created_data/cleaned_data/osiris.csv")

    return conn, emb, meta, model, E, O, R


conn, embeddings, meta, model, E, O, R = load_resources()
