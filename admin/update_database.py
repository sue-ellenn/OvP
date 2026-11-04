import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd


# Pas dit aan als jouw database op een andere locatie staat.
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "search.db"
BACKUP_DIR = BASE_DIR / "data" / "backups"


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def update_osiris(df):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database niet gevonden: {DB_PATH}")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"database_backup_{timestamp}.db"

    updated = 0
    inserted = 0

    conn = sqlite3.connect(DB_PATH)

    try:
        # Veilige databaseback-up
        backup_conn = sqlite3.connect(backup_path)
        try:
            conn.backup(backup_conn)
        finally:
            backup_conn.close()

        with conn:
            # Controleer of de tabel bestaat
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='osiris'"
            ).fetchone()

            if not table:
                raise ValueError("De tabel 'osiris' bestaat niet.")

            # Lees de bestaande kolommen
            columns = conn.execute(
                "PRAGMA table_info(osiris)"
            ).fetchall()

            db_columns = {col[1].casefold(): col[1] for col in columns}

            if "cursus" not in db_columns:
                raise ValueError("De kolom CURSUS ontbreekt in de database.")

            key_column = db_columns["cursus"]

            # Controleer bestaande dubbele cursuscodes
            duplicates = conn.execute(
                f"""
                SELECT {quote_identifier(key_column)}, COUNT(*)
                FROM osiris
                GROUP BY {quote_identifier(key_column)}
                HAVING COUNT(*) > 1
                LIMIT 10
                """
            ).fetchall()

            if duplicates:
                raise ValueError(
                    f"Dubbele cursuscodes in bestaande database: {duplicates}"
                )

            # Koppel Excel-kolommen aan databasekolommen
            column_mapping = {
                col: db_columns[col.casefold()]
                for col in df.columns
                if col.casefold() in db_columns
            }

            if "CURSUS" not in column_mapping:
                raise ValueError("CURSUS kan niet gekoppeld worden.")

            key_column = column_mapping["CURSUS"]

            update_columns = [
                column for column in column_mapping.values()
                if column != key_column
            ]

            if not update_columns:
                raise ValueError("Er zijn geen gegevenskolommen om bij te werken.")

            select_sql = (
                f"SELECT 1 FROM osiris WHERE {quote_identifier(key_column)} = ?"
            )

            update_sql = (
                f"UPDATE osiris SET "
                + ", ".join(
                    f"{quote_identifier(col)} = ?" for col in update_columns
                )
                + f" WHERE {quote_identifier(key_column)} = ?"
            )

            insert_columns = list(column_mapping.values())

            insert_sql = (
                f"INSERT INTO osiris ("
                + ", ".join(quote_identifier(col) for col in insert_columns)
                + ") VALUES ("
                + ", ".join("?" for _ in insert_columns)
                + ")"
            )

            for _, row in df.iterrows():
                course_code = str(row["CURSUS"]).strip()

                values = {
                    column_mapping[col]: (
                        None if pd.isna(row[col]) else str(row[col])
                    )
                    for col in column_mapping
                }

                existing = conn.execute(
                    select_sql, (course_code,)
                ).fetchone()

                if existing:
                    update_values = [
                        values[col] for col in update_columns
                    ]
                    conn.execute(
                        update_sql,
                        update_values + [course_code]
                    )
                    updated += 1

                else:
                    conn.execute(
                        insert_sql,
                        [values[col] for col in insert_columns]
                    )
                    inserted += 1

        return {
            "updated": updated,
            "inserted": inserted,
            "backup": str(backup_path)
        }

    finally:
        conn.close()