import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = "db/search.db"
OUTPUT_DIR = Path("db_csv")

TABLES = [
    "employee",
    "osiris",
    "repo",
    "course_employee",
]

OUTPUT_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)

for table in TABLES:
    df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    output = OUTPUT_DIR / f"{table}.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"{table}: {len(df):,} rows → {output}")

conn.close()

print("Done.")
