
import sqlite3
import shutil
from datetime import datetime
import sys
from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent))
from build_index import build_index

import pandas as pd

from scrapers.emp_scraper import scrape_all_employees
from scrapers.repo_scraper import scrape_full_repo


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "search.db"
BACKUP_DIR = BASE_DIR / "data" / "backups"


def create_backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"search_{datetime.now():%Y%m%d_%H%M%S}.db"
    shutil.copy2(DB_PATH, backup)
    print(f"Backup gemaakt: {backup}")


def validate_data(employees, repo):
    if employees.empty or not {"Name", "Url"}.issubset(employees.columns):
        raise ValueError("Employee-data ontbreekt of mist Name/Url.")

    if repo.empty or not {"title", "title_url", "department"}.issubset(repo.columns):
        raise ValueError("Repo-data ontbreekt of mist verplichte kolommen.")

    employees = employees.drop_duplicates(subset=["Url"])
    repo = repo.drop_duplicates(subset=["title_url"])

    return employees, repo


def clean_value(value):
    if isinstance(value, (list, tuple, dict)):
        import json
        return json.dumps(value, ensure_ascii=False)

    if value is None:
        return None

    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return None

    return str(value)


def update_table(conn, table, df, mapping, key):
    columns = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})")
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
            update_data = {k: v for k, v in data.items() if k != key}

            if update_data:
                assignments = ", ".join(f'"{k}" = ?' for k in update_data)
                conn.execute(
                    f'UPDATE "{table}" SET {assignments} WHERE rowid = ?',
                    (*update_data.values(), existing[0])
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


def update_database(employees, repo):
    with sqlite3.connect(DB_PATH) as conn:
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

        employee_counts = update_table(
            conn, "employee", employees, employee_mapping, "url"
        )

        repo_counts = update_table(
            conn, "repo", repo, repo_mapping, "title_url"
        )

    return employee_counts, repo_counts


def run_update():
    print("Start employee scraper...")
    employees = scrape_all_employees()
    print(f"Employees gevonden: {len(employees)}")

    print("\nStart Repo scraper...")
    repo = scrape_full_repo()
    print(f"Publicaties gevonden: {len(repo)}")

    employees, repo = validate_data(employees, repo)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database niet gevonden: {DB_PATH}")

    create_backup()

    employee_counts, repo_counts = update_database(employees, repo)

    print("\nDatabase bijgewerkt!")
    print("Employees:", employee_counts)
    print("Repo:", repo_counts)

    print("\nZoekindex en embeddings opnieuw opbouwen...")
    build_index()

    print("\nVolledige data-update afgerond!")


if __name__ == "__main__":
    run_update()
