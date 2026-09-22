import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

conn = sqlite3.connect("data/search.db")
c = conn.cursor()

c.execute(
    """
    DROP TABLE IF EXISTS search
    """
)

c.execute("""
CREATE VIRTUAL TABLE search
USING fts5(name, source, text)
""")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings = []
meta = []


def build_search_text(row):
    parts = []
    columns = {str(col).lower(): col for col in row.index}

    for col, weight in [
        ("keywords", 4),
        ("onderzoeksthema", 4),
        ("onderzoeksgroep", 3),
        ("doel", 4),
        ("inhoud", 4)
    ]:
        actual_col = columns.get(col)
        if actual_col is not None and pd.notna(row[actual_col]):
            value = str(row[actual_col]).strip()
            if value:
                parts.append((value + " ") * weight)

    return " ".join(parts)


def process(df, name_col, source):
    print("--------------------------")

    for _, row in df.iterrows():
        name = str(row[name_col]) if pd.notna(row[name_col]) else ""
        text = name + " " + build_search_text(row)

        c.execute(
            "INSERT INTO search VALUES (?, ?, ?)",
            (name, source, text)
        )

        print("Processed", name)

        embeddings.append(model.encode(text))
        meta.append({
            "name": name,
            "source": source,
            "index": int(row.name)
        })

R = pd.read_csv("created_data/cleaned_data/repo.csv")
E = pd.read_csv("created_data/cleaned_data/employee.csv")
O = pd.read_csv("created_data/cleaned_data/osiris.csv")

process(O, "CURSUS", "Osiris")
print("osiris done!")
process(E, "Name", "Employees")
print("employees done!")
process(R, "title", "Repo")
print("repo done!")

print(conn.execute("""
    SELECT name, source, text
    FROM search
    WHERE source = 'Employees'
    LIMIT 3
""").fetchall())

keyword = "neuromorphic"

print(conn.execute("""
    SELECT name, source, text
    FROM search
    WHERE source = 'Employees'
    AND text LIKE ?
""", (f"%{keyword}%",)).fetchall())

conn.commit()
conn.close()

np.save("embeddings.npy", np.array(embeddings))
np.save("meta.npy", np.array(meta, dtype=object))
