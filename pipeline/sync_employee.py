import sqlite3
from db.employee_index import create_employee_index, get_employee_index, update_employee_index
from scraper.sc_utils import content_hash

DB_PATH = "search.db"


def get_employee_columns(conn):
    return [row[1] for row in conn.execute("PRAGMA table_info(employee)").fetchall()]


def employee_exists(conn, url):
    return conn.execute("SELECT 1 FROM employee WHERE Url=? LIMIT 1",
                        (url,)).fetchone() is not None


def insert_employee(conn, data):
    columns = get_employee_columns(conn)
    data = {k: v for k, v in data.items() if k in columns}
    columns = list(data.keys())
    placeholders = ",".join(["?"] * len(columns))
    conn.execute(f"INSERT INTO employee ({','.join(columns)}) VALUES ({placeholders})", [data[c] for c in columns])


def update_employee(conn, data):
    columns = get_employee_columns(conn)
    data = {k: v for k, v in data.items() if k in columns}
    url = data.pop("Url")
    assignments = ",".join(f"{column}=?" for column in data)
    conn.execute(f"UPDATE employee SET {assignments} WHERE Url=?", [data[column] for column in data] + [url])


def sync_employee_data(scraped_df):
    create_employee_index()
    existing = get_employee_index()
    conn = sqlite3.connect(DB_PATH)
    added = 0
    updated = 0

    try:
        for _, row in scraped_df.iterrows():
            data = row.to_dict()
            url = data["Url"]
            name = data["Name"]

            if url in existing:
                if existing[url]["content_hash"] == data["_hash"]:
                    update_employee_index(url, name, data["_hash"])
                    continue

                update_employee(conn, data)
                updated += 1

            else:
                insert_employee(conn, data)
                added += 1

            update_employee_index(url, name, data["_hash"])

        conn.commit()
    finally:
        conn.close()

    print(f"Employees toegevoegd: {added}")
    print(f"Employees bijgewerkt: {updated}")
