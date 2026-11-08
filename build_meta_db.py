import sqlite3
import pandas as pd

DATABASE = "search.db"

EMPLOYEE_PATH = "created_data/cleaned_data/employee.csv"
OSIRIS_PATH = "created_data/cleaned_data/osiris.csv"
REPO_PATH = "created_data/cleaned_data/repo.parquet"


def create_tables(conn):
    cur = conn.cursor()

    # medewerkers

    cur.execute("""
    DROP TABLE IF EXISTS employee
    """)

    cur.execute("""
    CREATE TABLE employee(
        id INTEGER PRIMARY KEY,
        name TEXT,
        url TEXT,
        faculties TEXT,
        keywords TEXT,
        onderzoeksthema TEXT,
        onderzoeksgroep TEXT,
        publicaties TEXT,
        onderwijs TEXT
    )
    """)

    # osiris

    cur.execute("""
    DROP TABLE IF EXISTS osiris
    """)

    cur.execute("""
    CREATE TABLE osiris(
        id INTEGER PRIMARY KEY,
        cursus TEXT,
        lange_naam TEXT,
        docent_rol TEXT,
        doel TEXT,
        inhoud TEXT
    )
    """)

    # repository

    cur.execute("""
    DROP TABLE IF EXISTS repo
    """)

    cur.execute("""
    CREATE TABLE repo(
        id INTEGER PRIMARY KEY,
        title TEXT,
        authors TEXT,
        department TEXT,
        keywords TEXT,
        title_url TEXT,
        author_urls TEXT,
        publishing_info TEXT
    )
    """)

    conn.commit()


def load_employee(conn):
    print("Loading employees...")

    df = pd.read_csv(
        EMPLOYEE_PATH
    )

    rows = []

    for _, row in df.iterrows():
        rows.append(
            (
                row.get("Name"),
                row.get("Url"),
                row.get("Faculties"),
                row.get("Keywords"),
                row.get("Onderzoeksthema"),
                row.get("Onderzoeksgroep"),
                row.get("Publicaties"),
                row.get("Onderwijs")
            )
        )

    conn.executemany(

        """
        INSERT INTO employee(

            name,
            url,
            faculties,
            keywords,
            onderzoeksthema,
            onderzoeksgroep,
            publicaties,
            onderwijs

        )

        VALUES (?,?,?,?,?,?,?,?)

        """,
        rows)

    conn.commit()

    print("Employees:", len(rows))


def load_osiris(conn):
    print("Loading osiris...")

    df = pd.read_csv(OSIRIS_PATH)

    rows = []

    for _, row in df.iterrows():
        rows.append(
            (
                row.get("CURSUS"),
                row.get("LANGE_NAAM_NL"),
                row.get("DOCENT_ROL"),
                row.get("DOEL"),
                row.get("INHOUD")
            )
        )

    conn.executemany(

        """
        INSERT INTO osiris(

            cursus,
            lange_naam,
            docent_rol,
            doel,
            inhoud

        )

        VALUES (?,?,?,?,?)

        """,
        rows)

    conn.commit()

    print("Osiris:", len(rows))


def load_repo(conn):
    print("Loading repository...")

    df = pd.read_parquet(REPO_PATH)

    rows = []

    for _, row in df.iterrows():
        rows.append(

            (
                row.get("title"),
                row.get("authors"),
                row.get("department"),
                row.get("keywords"),
                row.get("title_url"),
                row.get("author_urls"),
                row.get("publishing_info")
            )
        )

    conn.executemany(

        """
        INSERT INTO repo(

            title,
            authors,
            department,
            keywords,
            title_url,
            author_urls,
            publishing_info

        )

        VALUES (?,?,?,?,?,?,?)

        """,
        rows)

    conn.commit()

    print("Repo:", len(rows))


def create_indexes(conn):
    print("Creating indexes...")

    cur = conn.cursor()

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_employee_name

    ON employee(name)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_osiris_course
    ON osiris(cursus)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_repo_title

    ON repo(title)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_course_employee_course

    ON course_employee(course)
    """)

    conn.commit()


conn = sqlite3.connect(
    DATABASE
)

create_tables(conn)
load_employee(conn)
load_osiris(conn)
load_repo(conn)
create_indexes(conn)
conn.close()

print("Database finished!")
