import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB="database/search.db"

MODEL="paraphrase-multilingual-MiniLM-L12-v2"

def update_embeddings():
    conn=sqlite3.connect(DB)
    df=np.array(conn.execute("SELECT name,Keywords FROM employee").fetchall())
    model=SentenceTransformer(MODEL)
    embeddings=model.encode([" ".join(map(str,x)) for x in df],normalize_embeddings=True)
    np.save("embeddings.npy",embeddings)
    np.save("meta.npy",df)
    print(f"Embeddings gemaakt: {len(df)}")

if __name__=="__main__":
    update_embeddings()