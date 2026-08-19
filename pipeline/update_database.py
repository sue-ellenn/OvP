from scrapers.emp_scraper import scrape_employee, scrape_all_employees
from db.db import get_connection


def update_database():
    df = scrape_all_employees()
    conn = get_connection()
    df.to_sql("employee", conn, if_exists="append", index=False)  # if_exists="replace
    conn.close()
    print(f"Employee update: {len(df)} records")


if __name__ == "__main__":
    update_database()
