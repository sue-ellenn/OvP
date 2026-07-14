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
from utils import *

# pd.read_csv("created_data/cleaned_data/repo.csv").to_parquet('created_data/cleaned_data/repo.parquet', compression="snappy")

FILES = {
    "search.db": "https://github.com/sue-ellenn/OvP/releases/download/data/search.db",
    "embeddings.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy",
    "meta.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/meta.npy"
}
# https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = "search.db"
EMBEDDINGS_PATH = FILES["embeddings.npy"]

response = requests.get(EMBEDDINGS_PATH)
response.raise_for_status()

embeddings = np.load(BytesIO(response.content))
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


# conn, embeddings, meta, model, E, O, R = load_resources()


def expand_query(query, model, embeddings, meta):
    expansions = [query]

    # alleen doen bij korte/vage queries
    if len(query) <= 5:
        similar = get_similar_terms(query, model, embeddings, meta, top_k=5)
        expansions.extend(similar)

    return list(set(expansions))


def expand_query_with_user_input(query, selected_terms):
    return " ".join([query] + selected_terms)


def get_query_suggestions(query, meta, embeddings, model, top_k=10):
    q_emb = model.encode(query)

    sims = np.dot(embeddings, q_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    )

    top_idx = np.argsort(sims)[-top_k:][::-1]

    suggestions = list({
        str(meta[i]["name"]) for i in top_idx
        if "name" in meta[i]
    })

    return suggestions


def get_suggestions_from_results(df):
    words = []

    for name in df["name"].head(20):
        words.extend(str(name).lower().split())

    words = [w for w in words if len(w) > 4]

    return list(set(words))[:10]


def get_similar_terms(query, model, embeddings, meta, top_k=5):
    q_emb = model.encode(query)
    q_emb = q_emb / np.linalg.norm(q_emb)

    sims = np.dot(embeddings, q_emb)

    top_idx = np.argsort(sims)[-top_k:][::-1]

    similar_terms = []
    for i in top_idx:
        text = str(meta[i])

        # pak eerste paar woorden als representatie
        term = text.split(" ")[:3]
        similar_terms.append(" ".join(term))

    return list(set(similar_terms))


def run_search(query, conn, embeddings, meta, model, TOP_FTS):
    fts_query = build_fts_query(query)

    sources = ["Employees", "Osiris", "Repo"]
    dfs = []

    expanded_queries = expand_query(query, model, embeddings, meta)
    with st.expander("**Actual query:**"):
        st.markdown(f"{expanded_queries}")
    query_embeddings = [
        model.encode(q) for q in expanded_queries
    ]

    # normaliseren (belangrijk!)
    query_embeddings = [
        q / np.linalg.norm(q) for q in query_embeddings
    ]

    # def max_sim(rowid):
    #     doc_emb = embeddings[rowid - 1]
    #     return max(
    #         cosine_sim(q_emb, doc_emb)
    #         for q_emb in query_embeddings
    #     )

    def max_sim(rowid):
        doc_emb = embeddings[rowid - 1]

        scores = []
        for q, q_emb in zip(expanded_queries, query_embeddings):
            score = cosine_sim(q_emb, doc_emb)

            # boost originele query
            if q == query:
                score *= 1.2

            scores.append(score)

        return max(scores)

    for source in sources:
        df = pd.read_sql_query(
            """
            SELECT rowid, name, source, bm25(search) AS rank
            FROM search
            WHERE search MATCH ?
            AND source = ?
            ORDER BY rank
            LIMIT ?
            """,
            conn,
            params=(fts_query, source, TOP_FTS)
        )

        if df.empty:
            continue

        df["bm25_score"] = 1 / (1 + df["rank"])

        # df["semantic_score"] = [
        #     cosine_sim(q_emb, embeddings[rowid - 1])
        #     for rowid in df["rowid"]
        # ]

        df["semantic_score"] = [
            max_sim(rowid)
            for rowid in df["rowid"]
        ]

        df["final_score"] = (
                0.6 * df["semantic_score"] +
                0.4 * df["bm25_score"]
        )

        # 🔥 per bron top N pakken
        df = df.sort_values("final_score", ascending=False).head(20)

        dfs.append(df)

    return pd.concat(dfs)
