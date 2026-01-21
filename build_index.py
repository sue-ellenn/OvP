import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


conn = sqlite3.connect("search.db")
c = conn.cursor()

c.execute("""
CREATE VIRTUAL TABLE search
USING fts5(name, source, text)
""")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = []
meta = []

def build_search_text(row):
    parts = []
    if "keywords" in row and pd.notna(row["keywords"]):
        parts.append((row["keywords"] + " ") * 5)
    if "Onderzoeksthema" in row and pd.notna(row["Onderzoeksthema"]):
        parts.append((row["Onderzoeksthema"] + " ") * 4)
    if "Onderzoeksgroep" in row and pd.notna(row["Onderzoeksgroep"]):
        parts.append((row["Onderzoeksgroep"] + " ") * 3)
    if "DOEL" in row and pd.notna(row["DOEL"]):
        parts.append((row["DOEL"] + " ") * 2)
    if "INHOUD" in row and pd.notna(row["INHOUD"]):
        parts.append((row["INHOUD"] + " ") * 2)

    return " ".join(parts)


def process(df, name_col, source):
    for _, row in df.iterrows():
        text = build_search_text(row)
        c.execute(
            "INSERT INTO search VALUES (?, ?, ?)",
            (row[name_col], source, text)
        )
        embeddings.append(model.encode(text))
        meta.append((row[name_col], source))

O = pd.read_csv("O.csv")
E = pd.read_csv("E.csv")
R = pd.read_csv("R.csv")

process(O, "CURSUS", "Osiris")
process(E, "Name", "Employees")
process(R, "title", "Repo")

conn.commit()
conn.close()

np.save("embeddings.npy", np.array(embeddings))
np.save("meta.npy", np.array(meta, dtype=object))
