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
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin

BASE_URL = "https://repository.ubn.ru.nl/"


def create_repo_session():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/128.0.0.0 Safari/537.36",
        "Referer": BASE_URL}
    session = requests.Session()

    response = session.get(BASE_URL, headers=headers)
    response.raise_for_status()
    print(f"Base url connected: {base_url}")
    # base_title = urljoin(BASE_URL, "browse?type=title")
    # response = session.get(base_title, headers=headers)
    # response.raise_for_status()

    return session, headers


def scrape_repo_departments(session, headers):
    base_title = "https://repository.ubn.ru.nl/browse?type=title"
    base_dep = "https://repository.ubn.ru.nl/browse?type=authorganizationcode"
    params = {
        "rpp": 50,
        "sort_by": -1,
        "type": "authorganizationcode",
        "etal": -1,
        "order": "ASC"
    }
    session = requests.Session()
    resp1 = session.get(base_url, headers=headers)
    resp1.raise_for_status()
    print(f"Base url connected2: {base_url}")

    # resp2 = requests.get(base_title, headers=headers)
    # resp2.raise_for_status()
    # print(f"Title url connected: {base_title}")

    # headers['referer'] = base_dep
    # time.sleep(2)
    # get actual url
    response = requests.get(base_dep, headers=headers)
    print(f"try dep: {base_dep}")
    response.raise_for_status()

    print(f"Dep url connected: {base_dep}")

    # response = session.get(base_dep, headers=headers)
    # response.raise_for_status()
    print(f"Dep url connected: {base_dep}")
    soup = BeautifulSoup(response.text, "html.parser")

    pagination = soup.find("p", class_="pagination-info")
    match = re.findall(r"(\d+)", pagination.get_text()) if pagination else []

    if not match:
        raise RuntimeError("Aantal departments kon niet worden gevonden.")

    total_results = int(match[-1])
    departments = []

    for offset in range(0, total_results, params["rpp"]):
        print(f"Department offset: {offset}/{total_results}")

        page_params = params.copy()
        page_params["offset"] = offset

        response = session.get(base_dep, headers=headers, params=page_params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for cell in soup.find_all("td", class_="ds-table-cell odd"):
            a_tag = cell.find("a")

            if not a_tag or not a_tag.get("href"):
                continue

            departments.append({"department": a_tag.get_text(strip=True), "url": urljoin(BASE_URL, a_tag["href"])})

        time.sleep(1)

    return pd.DataFrame(departments)


def scrape_repo(session, headers, target_url, department):
    params = {"rpp": 200, "sort_by": 1, "type": "title", "etal": -1, "order": "ASC"}

    # Belangrijk: target_url wordt met DEZELFDE session geopend.
    response = session.get(target_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    pagination = soup.find("p", class_="pagination-info")
    match = re.findall(r"(\d+)", pagination.get_text()) if pagination else []

    if not match:
        raise RuntimeError(f"Aantal Repo-resultaten kon niet worden gevonden voor {department}.")

    total_results = int(match[-1])
    data = []

    for offset in range(0, total_results, params["rpp"]):
        print(f"{department} - offset {offset}/{total_results}")

        page_params = params.copy()
        page_params["offset"] = offset

        response = session.get(target_url, headers=headers, params=page_params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        titles = soup.find_all("div", class_="artifact-description")

        for t in titles:
            title_tag = t.find("h4", class_="artifact-title")

            if not title_tag:
                continue

            a_tag = title_tag.find("a")
            title = a_tag.get_text(strip=True) if a_tag else title_tag.get_text(strip=True)
            title_url = urljoin(BASE_URL, a_tag["href"]) if a_tag and a_tag.get("href") else "N/A"

            authors = t.find_all("span", class_="ds-dc_contributor_author-authority-isRU")
            author_list = []
            author_urls = []

            for author in authors:
                link_tag = author.find("a")
                author_list.append(author.get_text(strip=True))

                if link_tag and link_tag.get("href"):
                    author_urls.append(urljoin(BASE_URL, link_tag["href"]))

            pub_tag = t.find("span", class_="content")
            publishing_info = pub_tag.get_text(strip=True) if pub_tag else "N/A"

            data.append({"title": title, "authors": author_list, "department": department, "keywords": "",
                         "title_url": title_url, "author_urls": author_urls, "publishing_info": publishing_info})

        time.sleep(1)

    return pd.DataFrame(data)


def scrape_full_repo():
    session, headers = create_repo_session()
    repo_departments = scrape_repo_departments(session, headers)

    print(f"Departments gevonden: {len(repo_departments)}")

    all_data = []

    for index, row in repo_departments.iterrows():
        print(f"{index}: {row['department']}")
        print(row["url"])

        df = scrape_repo(session, headers, row["url"], row["department"])

        if not df.empty:
            all_data.append(df)
        break

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def test_repo_access():
    base_url = BASE_URL
    base_dep = urljoin(base_url, "browse?type=authorganizationcode")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Referer": base_url}

    session = requests.Session()

    response = session.get(base_url, headers=headers)
    print("BASE:", response.status_code, response.url)
    print("COOKIES:", session.cookies.get_dict())
    response.raise_for_status()

    response = session.get(base_dep, headers=headers)
    print("DEPARTMENTS:", response.status_code, response.url)
    print("COOKIES:", session.cookies.get_dict())
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


# df = scrape_full_repo()
# print(df.shape)
# print(df.columns)
# print(df.head())
