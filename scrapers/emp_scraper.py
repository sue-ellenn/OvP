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

from scrapers.sc_utils import content_hash
from db.employee_index import create_employee_index, get_employee_index, update_employee_index

create_employee_index()

file_headers = ["Name", "Url", "Faculties", "Keywords", "Onderzoeksthema", "Onderzoeksgroep", "Publicaties",
                "Onderzoeksbeurzen en -prijzen", "Projecten", "Onderwijs", "In de media", "Curriculum Vitae",
                "Nevenwerkzaamheden"]

print()


def scrape_all_employees():
    playwright, browser, page = get_browser()

    try:
        employee_urls = get_employee_urls(page)
        employees = [parse_employee(page, name, url, file_headers) for name, url in employee_urls]
        df = pd.DataFrame(employees)
        print("Scraped columns:", list(df.columns))
        print("Employees scraped:", len(df))
        return df


    finally:
        browser.close()
        playwright.stop()


def get_employee_urls(page):
    base_url = "https://www.ru.nl"
    target = "https://www.ru.nl/zoeken/scope/medewerkers?w="

    all_employees = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/128.0.0.0 Safari/537.36",
        "Referer": target,
    }

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # create session from OG webpage
    soup = get_soup(page, target)
    # html, soup = get_page(page, target)

    TOTAL_RESULTS = soup.find(string=re.compile(r"\bResultaat\b", re.I))

    if not TOTAL_RESULTS:
        raise ValueError("Aantal employee-resultaten kon niet worden gevonden.")

    print("TOTAL_RESULTS", TOTAL_RESULTS)
    match_ = re.findall(r"(\d+)", TOTAL_RESULTS.string)
    TOTAL_RESULTS = int(match_[-1])
    RPP = int(match_[-2])
    print("RPP:", RPP)

    page_count = 0

    for offset in range(0, TOTAL_RESULTS, RPP):
        print("Offset", offset)

        employee_list = soup.find_all("h2", class_="card__title")

        for emp in employee_list:
            print("------------------------------------------------")

            # get name
            # name = emp.find("a")
            # try:
            #     emp_url = urljoin(base_url, name["href"])
            # except:
            #     continue
            #
            # print("URL: ", emp_url)
            #
            # name = emp.find("span", class_="link__text").string
            # print("Name", name)
            #
            # all_employees.append((name, emp_url))

            name_tag = emp.find("a")

            if not name_tag or not name_tag.get("href"):
                continue

            emp_url = urljoin(base_url, name_tag["href"])
            print("URL: ", emp_url)
            name_tag = emp.find("span", class_="link__text")
            name = name_tag.get_text(strip=True) if name_tag else None
            print("Name: ", name)
            if name:
                all_employees.append((name, emp_url))

        print("------------------------------------------------")

        # next page
        page_count += 1

        next_target = f"https://www.ru.nl/zoeken/scope/medewerkers?w=&page={page_count}"
        # next_target = f"target?w=&page={page_count}"
        soup = get_soup(page, next_target)
        print("Next page: succes!")
        break

    print("Finished")
    return all_employees
    # pass


# def extract_employee_page(page, url, file_headers):
#     page_soup = get_soup(page, url)
#
#     """
#     ["Name", "Faculteit", "Keywords", "url", "Onderzoeksthema", "Onderzoeksgroep", "Publicaties",
#      "Onderzoeksbeurzen en -prijzen", "Projecten", "Onderwijs", "In de media", "Curriculum Vitae",
#      "Nevenwerkzaamheden"] )
#     """
#
#     data_dict = {}
#     # faculty/affiliations names
#     faculty_names = page_soup.find("p", class_="text text--intro")
#
#     if faculty_names:
#
#         data_dict["Faculties"] = list(faculty_names.stripped_strings)
#     else:
#         data_dict["Faculties"] = "None found"
#
#     print("Faculty names:", data_dict["Faculties"])
#
#     # small_header_soup = page_soup.find_all("div", class_="profile__content")
#
#     # onderzoeksthemas
#     small_headers = page_soup.find_all("span", class_="label")
#
#     # print("Small headers: ", small_headers)
#
#     if small_headers:
#         for h in small_headers:
#             h_str = h.string
#             # print("string: ", h_str)
#
#             if h_str in file_headers:
#                 # print("header: ", h.find_next("ul", class_="list"))
#                 print("--------------")
#                 print("Header:", h_str)
#                 link_list = h.find_next("ul", class_="list")
#                 links = link_list.find_all("a")
#
#                 themas = []
#                 if links:
#                     for link in links:
#                         link_str = link.string
#                         themas.append((link_str, link['href']))
#                         print(link_str)
#                     # print("list:", themas)
#                     #     print(h_str)
#
#                     data_dict[h_str] = themas
#
#     keywords = page_soup.find_all("span", class_="meta-data__item")
#     if keywords:
#         kw_list = []
#         for kw in keywords:
#             kw_str = kw.string
#             kw_list.append(kw_str)
#         data_dict["Keywords"] = kw_list
#         print("Keywords:", data_dict["Keywords"])
#     big_headers = page_soup.find_all("h3", class_=["accordion-item"])
#     # print("Big headers: ", big_headers)
#
#     if big_headers:
#         for h in big_headers:
#             h_str = h.get_text(strip=True)
#             print("--------------")
#             print("Big Header:", h_str)
#
#             if h_str in file_headers:
#                 link_list = h.find_next("ul", class_="list")
#                 if link_list:
#                     papers = link_list.find_all("li")  # "span"
#                     # urls = link_list.find_all("a")
#
#                     themas = []
#                     if papers:
#                         for pap in papers:
#                             url = pap.find_next("a")
#                             tuptup = (pap.get_text(strip=True), url["href"])
#                             # print("Tuple: ", tuptup)
#                             themas.append(tuptup)
#                             print(tuptup)
#
#                         data_dict[h_str] = themas
#
#     return data_dict


def extract_employee_page(page_soup, file_headers):
    # page_soup = get_soup(page, url)
    # page_soup = get_soup(page, url)

    data_dict = {}

    """
        ["Name", "Faculteit", "Keywords", "url", "Onderzoeksthema", "Onderzoeksgroep", "Publicaties",
         "Onderzoeksbeurzen en -prijzen", "Projecten", "Onderwijs", "In de media", "Curriculum Vitae",
         "Nevenwerkzaamheden"] )
        """

    faculty_names = page_soup.find("p", class_="text text--intro")
    data_dict["Faculties"] = list(faculty_names.stripped_strings) if faculty_names else "None found"

    small_headers = page_soup.find_all("span", class_="label")

    for h in small_headers:
        h_str = h.get_text(strip=True)

        if h_str not in file_headers:
            continue

        link_list = h.find_next("ul", class_="list")
        links = link_list.find_all("a") if link_list else []
        themas = [(link.get_text(strip=True), link.get("href")) for link in links]
        data_dict[h_str] = themas

    keywords = page_soup.find_all("span", class_="meta-data__item")

    if keywords:
        data_dict["Keywords"] = [kw.get_text(strip=True) for kw in keywords]

    big_headers = page_soup.find_all("h3", class_=["accordion-item"])

    for h in big_headers:
        h_str = h.get_text(strip=True)

        if h_str not in file_headers:
            continue

        link_list = h.find_next("ul", class_="list")

        if not link_list:
            continue

        papers = link_list.find_all("li")
        themas = []

        for pap in papers:
            url_tag = pap.find_next("a")

            if not url_tag:
                continue

            themas.append((pap.get_text(strip=True), url_tag.get("href")))

        data_dict[h_str] = themas

    return data_dict


def parse_employee(page, name, url, file_headers):
    html, soup = get_page(page, url)
    data = extract_employee_page(soup, file_headers)
    data["Name"] = name
    data["Url"] = url
    data["_hash"] = content_hash(html)
    return data


def get_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        html = page.content()
        browser.close()
    return html


def get_browser():
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    return playwright, browser, page


def get_soup(page, url):
    page.goto(url, wait_until="networkidle")
    return BeautifulSoup(page.content(), "html.parser")


def get_page(page, url):
    page.goto(url, wait_until="networkidle")
    html = page.content()
    return html, BeautifulSoup(html, "html.parser")


# def get_soup(url):
#     return BeautifulSoup(get_html(url), "html.parser")


# URL = "https://www.ru.nl/personen/kwisthout-j"
#
#
# def scrape_employee():
#     df = pd.read_csv(URL)
#     return df


if __name__ == "__main__":
    # print(get_employee_urls())
    # print(scrape_all_employees())
    df = scrape_all_employees()
    print(df.shape)
    print(df[["Name", "Url"]].head())
    print(df.head(3))
    print("-------------------------------------")
    index = get_employee_index()
    print("Aantal geïndexeerde employees:", len(index))
