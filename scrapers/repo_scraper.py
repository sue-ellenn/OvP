import pandas as pd
# # importing libraries
import time
import requests
from bs4 import BeautifulSoup
import string
import re
# from requests_html import HTMLSession
from playwright.sync_api import sync_playwright
import csv
from urllib.parse import urljoin
from datetime import datetime, timedelta
import os
import glob
# from requests_html import HTMLSession

base_url = "https://repository.ubn.ru.nl"
base_dep = "https://repository.ubn.ru.nl/browse?type=authorganizationcode"
# target_url_p2 = "https://repository.ubn.ru.nl/browse?rpp=50&sort_by=1&type=title&offset=50&etal=-1&order=ASC"


def scrape_repo_departments(current_time):
    base_url = "base"
    base_dep = "base/browse?type=authorganizationcode"


    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/128.0.0.0 Safari/537.36",
        "Referer": base_url,
    }

    PARAMS = {
        "rpp": 50,
        "sort_by": -1,
        "type": "authorganizationcode",
        "etal": -1,
        "order": "ASC"
    }

    TOTAL_RESULTS = 0

    # create session from OG webpage
    session = requests.Session()
    resp1 = session.get(base_url, headers=headers)
    resp1.raise_for_status()

    # get actual url
    response = requests.get(base_dep, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # get total number of results form webpage
    TOTAL_RESULTS = soup.find("p", class_="pagination-info")
    match_ = re.findall(r"(\d+)", TOTAL_RESULTS.get_text())
    TOTAL_RESULTS = int(match_[-1])
    RPP = PARAMS["rpp"]

    offset_count = 0

    data = []
    # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # create employee file
    emp_file_name = "created_data/repository/emp_" + current_time + ".csv"
    with open(emp_file_name, "w") as emp_file:
        pass

    new_file_name = "created_data/repository/department_" + current_time + ".csv"

    with open(new_file_name, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, ["department", "url"])
        writer.writeheader()
        for offset in range(0, TOTAL_RESULTS, RPP):
            # if offset >= offset_count:

            print("Offset", offset)
            params = PARAMS.copy()
            params["offset"] = offset
            titles = soup.find_all("td", class_="ds-table-cell odd")
            print("Titles:", titles)

            for t in titles:

                print("--------------------------------------------------------------------")

                dep_title = t

                if dep_title:
                    a_tag = dep_title.find("a")
                    # print("Tag:", a_tag['href'])
                    dep_url = urljoin(base_url, a_tag["href"]) if dep_title and a_tag.get("href") else "N/A"
                    dep_text = dep_title.string
                    title = dep_text.strip() if dep_text else a_tag.get_text(strip=True) if a_tag else "Untitled"
                else:
                    title = "Untitled"
                    dep_url = "N/A"

                print(f"Department: {title} | URL: {dep_url}")
                d = {"department": title, "url": dep_url}
                writer.writerow(d)
                data.append((title, dep_url))

                print("--------------------------------------------------------------------")

            next_page = soup.find("a", class_="next-page-link")
            next_page_url = ""

            if next_page:
                next_page_url = urljoin(base_url, next_page["href"])
                print(f"Next page: {next_page_url}")

                # get actual url
                response = requests.get(next_page_url, params=params, headers=headers)
                response.raise_for_status()
                time.sleep(1)
                soup = BeautifulSoup(response.text, 'html.parser')
                print("Next page: succes!")
            else:
                print("Next page: N/A")
                break

    df = pd.DataFrame(data)

    # df.to_csv("created_data/dep_url.csv", index=False)
    return df
