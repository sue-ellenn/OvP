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

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings = []
meta = []

def build_search_text(row):
    parts = []
    if "keywords" in row and pd.notna(row["keywords"]):
        parts.append((row["keywords"] + " ") * 4)
    if "Onderzoeksthema" in row and pd.notna(row["Onderzoeksthema"]):
        parts.append((row["Onderzoeksthema"] + " ") * 4)
    if "Onderzoeksgroep" in row and pd.notna(row["Onderzoeksgroep"]):
        parts.append((row["Onderzoeksgroep"] + " ") * 3)
    if "DOEL" in row and pd.notna(row["DOEL"]):
        parts.append((row["DOEL"] + " ") * 4)
    if "INHOUD" in row and pd.notna(row["INHOUD"]):
        parts.append((row["INHOUD"] + " ") * 4)

    return " ".join(parts)


def process(df, name_col, source):
    print("--------------------------")
    for _, row in df.iterrows():
        print("--------------------------")
        text = build_search_text(row)
        c.execute(
            "INSERT INTO search VALUES (?, ?, ?)",
            (row[name_col], source, text)
        )
        print("Processed", row[name_col])
        embeddings.append(model.encode(text))
        meta.append((row[name_col], source))

R = pd.read_csv("created_data/cleaned_data/repo.csv")
E = pd.read_csv("created_data/cleaned_data/employee.csv")
O = pd.read_csv("created_data/cleaned_data/osiris.csv")

process(O, "CURSUS", "Osiris")
print("osiris done!")
process(E, "Name", "Employees")
print("employees done!")
process(R, "title", "Repo")
print("repo done!")

conn.commit()
conn.close()

np.save("created_data/DB_backups/embeddings.npy", np.array(embeddings))
np.save("meta.npy", np.array(meta, dtype=object))
