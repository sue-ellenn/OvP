import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scrapers.emp_scraper import scrape_employee, scrape_all_employees
from scrapers.sc_utils import *
from db.db import get_connection



def update_database():
    df = scrape_all_employees()
    conn = get_connection()
    df.to_sql("employee", conn, if_exists="append", index=False)  # if_exists="replace
    conn.close()
    print(f"Employee update: {len(df)} records")


if __name__ == "__main__":
    update_database()
