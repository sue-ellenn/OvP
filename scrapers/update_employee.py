import sqlite3
from OvP.scrapers.emp_scraper import scrape_all_employees

DB_PATH = "search.db"


def update_employee():
    print("Employee update gestart...")
    df = scrape_all_employees()

    print(f"Employees gevonden: {len(df)}")
    print("Kolommen:", list(df.columns))
    print("DataFrame:", df.shape)

    conn = sqlite3.connect(DB_PATH)
    print()
    print(conn.execute("""SELECT *
                    FROM employees.COLUMNS 
    """))
    try:
        df.to_sql("employee", conn, if_exists="append", index=False)
        conn.commit()
        print("Employee tabel succesvol vervangen.")
    finally:
        conn.close()
    return df


if __name__ == "__main__":
    update_employee()

# def update_employee():
#     print("Employee update gestart...")
#     df = scrape_all_employees()
#     print(f"Employees gevonden: {len(df)}")
#
#     conn = sqlite3.connect(DB_PATH)
#
#     try:
#         df.to_sql("employee", conn, if_exists="append", index=False)  # replace
#         conn.commit()
#         print("Employee tabel succesvol vervangen.")
#     finally:
#         conn.close()


if __name__ == "__main__":
    update_employee()
