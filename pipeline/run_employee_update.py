from scraper.emp_scraper import scrape_all_employees
from pipeline.sync_employee import sync_employee_data


def run_employee_update():
    print("Employee update gestart")
    df = scrape_all_employees()
    print(f"Employees gescrapet: {len(df)}")
    sync_employee_data(df)
    print("Employee update klaar")


if __name__ == "__main__":
    run_employee_update()
