
import sqlite3
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "search.db"
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 256


def value_to_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_search_text(row, source):
    parts = []
    columns = {str(col).lower(): col for col in row.index}

    # Gewogen zoekvelden
    weights = [
        ("keywords", 4),
        ("onderzoeksthema", 4),
        ("onderzoeksgroep", 3),
        ("doel", 4),
        ("inhoud", 4)
    ]

    for col, weight in weights:
        actual_col = columns.get(col)

        if actual_col is not None:
            value = value_to_text(row[actual_col])

            if value:
                parts.extend([value] * weight)

    # Aanvullende informatie per databron
    extra_fields = {
        "Employees": [
            "name", "faculties", "publicaties", "onderwijs"
        ],
        "Repo": [
            "title", "authors", "department", "publishing_info"
        ],
        "Osiris": [
            "cursus", "lange_naam_nl", "docent_rol"
        ]
    }

    for col in extra_fields[source]:
        actual_col = columns.get(col)

        if actual_col is not None:
            value = value_to_text(row[actual_col])

            if value:
                parts.append(value)

    return " ".join(parts)


def load_data(conn):
    tables = ["employee", "repo", "osiris"]

    for table in tables:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        ).fetchone()

        if not exists:
            raise ValueError(f"Database-tabel ontbreekt: {table}")

    employees = pd.read_sql_query("SELECT * FROM employee", conn)
    repo = pd.read_sql_query("SELECT * FROM repo", conn)
    osiris = pd.read_sql_query("SELECT * FROM osiris", conn)

    if employees.empty or repo.empty or osiris.empty:
        raise ValueError(
            "Een of meer databronnen zijn leeg. Index wordt niet bijgewerkt."
        )

    return [
        (employees, "name", "Employees"),
        (repo, "title", "Repo"),
        (osiris, "cursus", "Osiris")
    ]


def build_index():
    print("Database inlezen...")

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        datasets = load_data(conn)

    documents = []

    for df, name_col, source in datasets:
        if name_col not in df.columns:
            raise ValueError(
                f"Kolom '{name_col}' ontbreekt bij {source}."
            )

        print(f"{source}: {len(df)} records")

        for row_index, row in df.iterrows():
            name = value_to_text(row[name_col])

            if not name:
                continue

            text = name + " " + build_search_text(row, source)

            if text.strip():
                documents.append({
                    "name": name,
                    "source": source,
                    "text": text,
                    "index": int(row_index)
                })

    if not documents:
        raise ValueError("Geen documenten gevonden om te indexeren.")

    print(f"Totaal documenten: {len(documents)}")
    print("Model laden...")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = []

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]
        texts = [doc["text"] for doc in batch]

        batch_embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings.append(batch_embeddings)

        print(
            f"Embeddings: {min(start + BATCH_SIZE, len(documents))}"
            f"/{len(documents)}"
        )

    embeddings = np.vstack(embeddings)

    meta = np.array([
        {
            "name": doc["name"],
            "source": doc["source"],
            "index": doc["index"]
        }
        for doc in documents
    ], dtype=object)

    print("Tijdelijke bestanden opslaan...")

    embeddings_tmp = DATA_DIR / "embeddings_tmp.npy"
    meta_tmp = DATA_DIR / "meta_tmp.npy"

    np.save(embeddings_tmp, embeddings)
    np.save(meta_tmp, meta)

    print("FTS5-index opbouwen...")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DROP TABLE IF EXISTS search_new")

            conn.execute("""
                CREATE VIRTUAL TABLE search_new
                USING fts5(name, source, text)
            """)

            conn.executemany(
                "INSERT INTO search_new (name, source, text) VALUES (?, ?, ?)",
                [
                    (doc["name"], doc["source"], doc["text"])
                    for doc in documents
                ]
            )

            conn.execute("DROP TABLE IF EXISTS search")
            conn.execute("ALTER TABLE search_new RENAME TO search")
            conn.commit()

        os.replace(embeddings_tmp, DATA_DIR / "embeddings.npy")
        os.replace(meta_tmp, DATA_DIR / "meta.npy")

    except Exception:
        embeddings_tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
        raise

    print("\nZoekindex succesvol opgebouwd!")
    print(f"FTS5-documenten: {len(documents)}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Metadata: {len(meta)}")


if __name__ == "__main__":
    build_index()
