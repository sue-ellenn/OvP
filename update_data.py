
import sqlite3
import shutil
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_index import build_index


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "search.db"
BACKUP_DIR = BASE_DIR / "data" / "backups"


def create_backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"search_{datetime.now():%Y%m%d_%H%M%S}.db"
    shutil.copy2(DB_PATH, backup)
    print(f"Backup gemaakt: {backup}")



def safe_scrape(source):
    try:
        if source == "Employees":
            from scrapers.emp_scraper import scrape_all_employees
            df = scrape_all_employees()
            required = {"Name", "Url"}
            key = "Url"

        elif source == "Repository":
            from scrapers.repo_scraper import scrape_full_repo
            df = scrape_full_repo()
            required = {"title", "title_url", "department"}
            key = "title_url"

        else:
            raise ValueError(f"Onbekende databron: {source}")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Scraper heeft geen DataFrame teruggegeven.")

        if df.empty:
            raise ValueError("Geen gegevens ontvangen.")

        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Verplichte kolommen ontbreken: {missing}")

        df = df.drop_duplicates(subset=[key])

        print(f"{source}: {len(df)} records opgehaald.")
        return df

    except Exception as e:
        print(f"{source} overgeslagen door fout: {type(e).__name__}: {e}")
        return None

def clean_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)

    if value is None:
        return None

    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return None

    return str(value)


def update_table(conn, table, df, mapping, key):
    columns = {
        row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')
    }

    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        data = {
            db_col: clean_value(row.get(source_col))
            for source_col, db_col in mapping.items()
            if db_col in columns
        }

        if not data.get(key):
            continue

        existing = conn.execute(
            f'SELECT rowid FROM "{table}" WHERE "{key}" = ?',
            (data[key],)
        ).fetchone()

        if existing:
            update_values = {k: v for k, v in data.items() if k != key}

            if update_values:
                assignments = ", ".join(f'"{k}" = ?' for k in update_values)
                conn.execute(
                    f'UPDATE "{table}" SET {assignments} WHERE rowid = ?',
                    (*update_values.values(), existing[0])
                )

            updated += 1

        else:
            names = ", ".join(f'"{k}"' for k in data)
            placeholders = ", ".join("?" for _ in data)

            conn.execute(
                f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
                tuple(data.values())
            )

            inserted += 1

    return {"inserted": inserted, "updated": updated}



def update_database(employees=None, repo=None):
    employee_mapping = {
        "Name": "name",
        "Url": "url",
        "Faculties": "faculties",
        "Keywords": "keywords",
        "Onderzoeksthema": "onderzoeksthema",
        "Onderzoeksgroep": "onderzoeksgroep",
        "Publicaties": "publicaties",
        "Onderwijs": "onderwijs"
    }

    repo_mapping = {
        "title": "title",
        "authors": "authors",
        "department": "department",
        "keywords": "keywords",
        "title_url": "title_url",
        "author_urls": "author_urls",
        "publishing_info": "publishing_info"
    }

    sources = [
        ("Employees", "employee", employees, employee_mapping, "url"),
        ("Repository", "repo", repo, repo_mapping, "title_url")
    ]

    counts = {}
    conn = sqlite3.connect(DB_PATH)

    try:
        for name, table, df, mapping, key in sources:
            if df is None:
                counts[name] = {"status": "overgeslagen"}
                continue

            try:
                # Iedere databron krijgt een eigen transactie
                with conn:
                    counts[name] = update_table(conn, table, df, mapping, key)

                counts[name]["status"] = "bijgewerkt"

            except Exception as e:
                counts[name] = {"status": "fout", "error": str(e)}
                print(f"Database-update {name} mislukt: {e}")

    finally:
        conn.close()

    return counts

def run_update():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database niet gevonden: {DB_PATH}")

    print("\n========== START DATA-UPDATE ==========")

    # Beide scrapers onafhankelijk uitvoeren
    employees = safe_scrape("Employees")
    repo = safe_scrape("Repository")

    # Alleen doorgaan als minimaal één scraper data heeft opgehaald
    if employees is None and repo is None:
        print("\nGeen nieuwe data beschikbaar. Database blijft ongewijzigd.")
        return

    # Backup vóór databasewijzigingen
    create_backup()

    # Database bijwerken
    counts = update_database(employees, repo)

    print("\n========== DATABASE RESULTATEN ==========")

    for source, result in counts.items():
        print(f"{source}: {result}")

    # Zoekindex alleen opnieuw bouwen als minimaal één update gelukt is
    successful_updates = any(
        result.get("status") == "bijgewerkt"
        for result in counts.values()
    )

    if successful_updates:
        print("\nZoekindex en embeddings opnieuw opbouwen...")

        try:
            build_index()
            print("Zoekindex succesvol opnieuw opgebouwd!")

        except Exception as e:
            print(f"Fout bij opnieuw opbouwen zoekindex: {e}")

    else:
        print("\nGeen database-updates uitgevoerd. Zoekindex blijft ongewijzigd.")

    print("\n========== DATA-UPDATE AFGEROND ==========")


if __name__ == "__main__":
    run_update()
