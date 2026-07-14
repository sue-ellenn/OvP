import base64
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

# @st.cache_resource
# def load_repo():
#     if not REPO_PATH.exists():
#         with st.spinner("Downloading repository data..."):
#             r = requests.get(REPO_URL)
#             r.raise_for_status()
#             REPO_PATH.write_bytes(r.content)
#     return pd.read_parquet(REPO_PATH)

# return pd.read_parquet(REPO_PATH)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = "search.db"
EMBEDDINGS_PATH = FILES["embeddings.npy"]

response = requests.get(EMBEDDINGS_PATH)
response.raise_for_status()

embeddings = np.load(BytesIO(response.content))
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_fts_query(user_query: str) -> str:
    terms = user_query.strip().split()
    if len(terms) == 1:
        return terms[0]
    exact = f'"{user_query}"'
    and_query = " AND ".join(terms)
    or_query = " OR ".join(terms)

    return f"{exact} OR ({and_query}) OR ({or_query})"


def normalize_name(name):
    parts = name.lower().split()
    return {
        "initial": parts[0][0] if parts else "",
        "last": parts[-1] if parts else ""
    }

def normalize(v):
    return v / np.linalg.norm(v)


def docent_match(employee_name, docent_rol):
    if not isinstance(docent_rol, str):
        return False
    emp = normalize_name(employee_name)
    rol = docent_rol.lower()
    return emp["last"] in rol and emp["initial"] in rol


def get_themas(employee_row):
    themas = []
    for col in ["Keywords", "Onderzoeksthema", "Onderzoeksgroep"]:
        if col in employee_row and pd.notna(employee_row[col]):
            themas.extend(employee_row[col].split(","))
    return sorted(set(t.strip() for t in themas if len(t.strip()) > 2))


def get_publicaties(name, R, max_items=3):
    last_name = name.split()[-1].lower()
    pubs = R[R["authors"].str.lower().str.contains(last_name, na=False)]
    pubs = pubs.sort_values("publishing_info", ascending=False)
    return pubs[["title", "publishing_info", "title_url"]].head(max_items)


def get_docent_cursussen(name, O, max_items=3):
    mask = O["DOCENT_ROL"].apply(
        lambda x: docent_match(name, x)
    )
    return O[mask][["CURSUS", "LANGE_NAAM_NL", "DOEL"]].head(max_items)

def get_employees_for_course(course_code, conn):

    df = pd.read_sql_query(
        """
        SELECT employee
        FROM course_employee
        WHERE course = ?
        """,
        conn,
        params=(course_code,)
    )

    return df["employee"].tolist()

def get_osiris_course(course_code, O):
    row = O[O["CURSUS"] == course_code]
    return None if row.empty else row.iloc[0]


def get_repository_record(title, R):
    row = R[R["title"] == title]
    return None if row.empty else row.iloc[0]


def interleave(dfs, max_total):
    result = []
    pointers = [0] * len(dfs)

    while len(result) < max_total:
        added = False

        for i, df in enumerate(dfs):
            if pointers[i] < len(df):
                result.append(df.iloc[pointers[i]])
                pointers[i] += 1
                added = True

                if len(result) >= max_total:
                    break

        if not added:
            break

    return pd.DataFrame(result)

# streamlit run app2.py --server.runOnSave true
