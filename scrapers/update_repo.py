import sqlite3
from OvP.scrapers.repo_scraper import scrape_full_repo, test_repo_access

DB_PATH = "search.db"


def update_repo():
    print("Repo update gestart...")
    df = scrape_full_repo()

    print(f"Repo records gevonden: {len(df)}")
    print("Kolommen:", list(df.columns))
    print(df.head())
    return df
    # conn=sqlite3.connect(DB_PATH)
    #
    # try:
    #     df.to_sql("repo",conn,if_exists="replace",index=False)
    #     conn.commit()
    #     print("Repo tabel succesvol vervangen.")
    # finally:
    #     conn.close()


if __name__ == "__main__":
    # update_repo()
    soup = test_repo_access()
    print(soup.title)

    pass
