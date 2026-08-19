import sqlite3
from db.employee_index import create_employee_index, get_employee_index, update_employee_index
from scrapers.sc_utils import content_hash

DB_PATH = "search.db"


def debug_employee_table(conn):
    rows = conn.execute("PRAGMA table_info(employee)").fetchall()
    print("Employee database columns:", [row[1] for row in rows])


def get_employee_columns(conn):
    return [row[1] for row in conn.execute("PRAGMA table_info(employee)").fetchall()]


def employee_exists(conn, url):
    return conn.execute("SELECT 1 FROM employee WHERE Url=? LIMIT 1",
                        (url,)).fetchone() is not None


# def insert_employee(conn, data):
#     columns = get_employee_columns(conn)
#     data = {k: v for k, v in data.items() if k in columns and k != "id"}
#
#     if not data:
#         raise ValueError(f"Geen bruikbare employee-kolommen gevonden. Scraper-kolommen: {list(data.keys())}")
#
#     column_names = ",".join(f'"{column}"' for column in data)
#     placeholders = ",".join("?" for _ in data)
#     values = list(data.values())
#     query = f'INSERT INTO employee ({column_names}) VALUES ({placeholders})'
#     conn.execute(query, values)

def insert_employee(conn, data):
    columns = get_employee_columns(conn)
    print("DB columns:", columns)
    print("Scraper columns:", list(data.keys()))

    valid_data = {k: v for k, v in data.items() if k in columns and k != "id"}
    print("Matching columns:", list(valid_data.keys()))

    if not valid_data:
        raise ValueError("Geen overlap tussen scraper- en databasekolommen.")

    column_names = ",".join(f'"{column}"' for column in valid_data)
    placeholders = ",".join("?" for _ in valid_data)
    values = list(valid_data.values())
    query = f'INSERT INTO employee ({column_names}) VALUES ({placeholders})'
    conn.execute(query, values)


def update_employee(conn, data):
    columns = get_employee_columns(conn)
    url = data.get("Url")

    if not url:
        raise ValueError("Employee heeft geen Url.")

    data = {k: v for k, v in data.items() if k in columns and k not in ("id", "Url")}

    if not data:
        return

    assignments = ",".join(f'"{column}"=?' for column in data)
    values = list(data.values())
    values.append(url)
    query = f'UPDATE employee SET {assignments} WHERE "Url"=?'
    conn.execute(query, values)


def sync_employee_data(scraped_df):
    create_employee_index()
    existing = get_employee_index()
    conn = sqlite3.connect(DB_PATH)
    debug_employee_table(conn)
    added = 0
    updated = 0

    try:
        for _, row in scraped_df.iterrows():
            data = row.to_dict()
            print("First row keys:", list(data.keys()))

            if "Url" not in data or "Name" not in data:
                print("Employee overgeslagen: Name of Url ontbreekt:", data.keys())
                continue

            url = data["Url"]
            name = data["Name"]

            if url in existing:
                if existing[url]["content_hash"] == data.get("_hash"):
                    update_employee_index(url, name, data.get("_hash"))
                    continue

                update_employee(conn, data)
                updated += 1
            else:
                insert_employee(conn, data)
                added += 1

            update_employee_index(url, name, data.get("_hash"))

        conn.commit()
    finally:
        conn.close()

    print(f"Employees toegevoegd: {added}")
    print(f"Employees bijgewerkt: {updated}")
