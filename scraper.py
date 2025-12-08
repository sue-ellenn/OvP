# # importing libraries
import time

import pandas as pd
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


def load_osiris_data(filename):
    df = pd.read_excel(filename)
    df.dropna(axis=0, how='any', subset=['INHOUD'], inplace=True)

    df['DOCENT_ROL'] = df.apply(format_teachers, axis=1)
    return df


def format_teachers(row):
    teachers = row['DOCENT_ROL']
    # 'Brasjen DOCENT; Bruinsma DOCENT; Buiks DOCENT; Burgh DOCENT; Galipò DOCENT; Gronden DOCENT; Kaathoven CONTACTPERSOON; Verhagen DOCENT'

    formatted_list = []

    teachers = teachers.split(";")
    roles = ['DOCENT', 'TEACHER', 'EXAMINATOR', 'EXAMINER']

    for teach in teachers:
        t = teach.split()

        if t[-1] in roles:
            formatted_list.extend(t[:-1])

    delimiter = "; "  # Define a delimiter
    join_str = delimiter.join(formatted_list)
    return join_str


def get_column_list(df, col):
    unique_names = (
        df[col]
        .str.split(';')  # split strings into lists
        .explode()  # make one name per row
        .str.strip()  # remove extra spaces (if any)
        .unique()  # get unique names
    )

    unique_names = list(unique_names)  # convert to a Python list if needed
    return unique_names


# def scrape_teachers(last_name, initial):
#     """
#     Scrape teachers data from RU website
#     https://www.ru.nl/zoeken/scope/medewerkers?w=
#
#     :param last_name:
#     :param initial:
#     :return:
#     """
#
#     # TODO
#     # collect headers
#     headers = []
#     headers = ['Name'].extend(headers)
#
#     teachers = pd.DataFrame(columns=headers)
#     #https://realpython.com/beautiful-soup-web-scraper-python/
#
#     # Radboud web zoekfunctie
#     # https://www.ru.nl/zoeken?w=
#     # https://www.ru.nl/zoeken/scope/medewerkers?w=
#
#     last_name, initial = last_name.lower().strip(), initial.lower().strip()
#     # print(f'Name: {last_name}, {initial}')
#
#     url = f"https://www.ru.nl/personen/{last_name}-{initial}"
#     # print(f'URL: {url}')
#     # url = f"https://www.ru.nl/personen/brasjen-m"
#
#     url =f"https://www.ru.nl/personen/kwisthout-j"
#
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     print("soup")
#
#     # TODO
#     # Collect every small header (automatically)
#     headers = ['Onderzoeksthema', 'Onderzoeksgroep']
#
#     # Collect every dropdown menu with content (automatically)
#
#     expertise_header = soup.find("span", class_="label", string="Onderzoeksthema")
#
#     container = expertise_header.find_next_sibling()
#     all_text = ";".join(container.stripped_strings)
#     print(f' Name: {last_name}, {initial}')
#     print(all_text)
#
#     return

def scrape_teachers():
    """
    Scrape teachers data from RU website
    https://www.ru.nl/zoeken/scope/medewerkers?w=

    :return:
    """

    # TODO
    # collect headers

    teachers = pd.DataFrame(columns=['Name'])
    #https://realpython.com/beautiful-soup-web-scraper-python/

    url = "https://www.ru.nl/zoeken/scope/medewerkers?w="
    #
    response = requests.get(url)
    print(response.text)
    soup = BeautifulSoup(response.text, 'html.parser')
    print("soup")
    # session = HTMLSession()
    #
    # # Load the page and render JavaScript
    # r = session.get(url)
    # r.html.render(timeout=10)  # let JS load
    #
    # # Parse rendered HTML
    # soup = BeautifulSoup(r.html.html, "html.parser")

    nr_pattern = re.compile(r"(\d+)\s*[-–]\s*(\d+)[^\d]+(\d+)", re.UNICODE)
    headers = soup.find_all("h2", class_="card__title")

    for h2 in headers:
        name = h2.get_text(strip=True)
        link = h2.find("a")["href"] if h2.find("a") else None
        print(f"Name: {name} | Link: {link}")

    # print(headers)
    # total_results = None
    #
    # for h in headers:
    #     text = h.get_text()
    #     print("text: ", text)
    #     matches = nr_pattern.search(text)
    #     if matches:
    #         total_results = int(matches.group(3))
    #         print("Matched header text:", text)
    #         break
    #
    #
    # print(total_results)
    # employee = soup.find("span", class_="label", string="Onderzoeksthema")
    #
    # # TODO
    # # Collect every small header (automatically)
    # headers = ['Onderzoeksthema', 'Onderzoeksgroep']
    #
    # # Collect every dropdown menu with content (automatically)
    #
    # expertise_header = soup.find("span", class_="label", string="Onderzoeksthema")
    #
    # container = expertise_header.find_next_sibling()
    # all_text = ";".join(container.stripped_strings)
    # print(f' Name: {last_name}, {initial}')
    # print(all_text)

    return


def scrape_repo(target_url, department, file):
    base_url = "https://repository.ubn.ru.nl"
    base_title = "https://repository.ubn.ru.nl/browse?type=title"
    # target_url = "https://repository.ubn.ru.nl/browse?rpp=200&type=title&resetOffset=true"
    # target_url_p2 = "https://repository.ubn.ru.nl/browse?rpp=50&sort_by=1&type=title&offset=50&etal=-1&order=ASC"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/128.0.0.0 Safari/537.36",
        "Referer": base_url,
    }

    PARAMS = {
        "rpp": 200,
        "sort_by": 1,
        "type": "title",
        "etal": -1,
        "order": "ASC"
    }

    TOTAL_RESULTS = 0

    # create session from OG webpage
    session = requests.Session()
    resp1 = session.get(base_url, headers=headers)
    resp1.raise_for_status()

    # go to title section
    resp2 = requests.get(base_title, headers=headers)
    resp2.raise_for_status()

    # get actual url
    response = requests.get(target_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # get total number of results form webpage
    TOTAL_RESULTS = soup.find("p", class_="pagination-info")
    match_ = re.findall(r"(\d+)", TOTAL_RESULTS.get_text())
    TOTAL_RESULTS = int(match_[-1])
    RPP = PARAMS["rpp"]

    offset_count = 0
    # quit()
    # new_file_name = "created_data/repo_data_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"

    data = []
    with open(file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, ["title", "authors", "department", "keywords", "title_url", "author_urls", "publishing_info"])
        writer.writeheader()

        for offset in range(0, TOTAL_RESULTS, RPP):
            # if offset >= offset_count:

            print("Offset", offset)
            params = PARAMS.copy()
            params["offset"] = offset
            titles = soup.find_all("div", class_="artifact-description")

            for t in titles:

                print("--------------------------------------------------------------------")

                # Title
                title_tag = t.find("h4", class_="artifact-title")
                if title_tag:
                    a_tag = title_tag.find("a")
                    title_url = urljoin(base_url, a_tag["href"]) if a_tag and a_tag.get("href") else "N/A"

                    title_text = title_tag.string
                    title = title_text.strip() if title_text else a_tag.get_text(strip=True) if a_tag else "Untitled"
                else:
                    title = "Untitled"
                    title_url = "N/A"

                print(f"Title: {title} | URL: {title_url}")

                #  Authors
                authors = t.find_all("span", class_="ds-dc_contributor_author-authority-isRU")
                author_list = []
                author_hrefs = []
                if authors:
                    for a in authors:
                        link_tag = a.find("a")
                        author_name = a.string
                        author_href = urljoin(base_url, link_tag["href"]) if link_tag and link_tag.get("href") else "N/A"
                        author_list.append(author_name)
                        author_hrefs.append(author_href)
                    # print nicely
                    print("Authors:")
                    for i in range(len(author_hrefs)):
                        print(f"  - {author_list[i]} | {author_hrefs[i]}")
                else:
                    print("Authors: None found")

                # Publication info
                pub_tag = t.find("span", class_="content")
                pub_text = pub_tag.get_text(strip=True) if pub_tag else "N/A"
                print(f"Publication: {pub_text}")

                print(f"Department: {department}")

                dict = {
                    "title": title,
                    "authors": author_list,
                    "department": department,  # to be filled
                    "keywords": "",  # empty for now
                    "title_url": title_url,
                    "author_urls": author_hrefs,
                    "publishing_info": pub_text
                }
                data.append(dict)
                writer.writerow(dict)
                f.flush()

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
    # with sync_playwright() as p:
    #     browser = p.chromium.launch(headless=True)
    #     page = browser.new_page()
    #     page.goto(base_url)
    #     page.wait_for_load_state("networkidle")  # wait for JS to finish
    #     html = page.content()
    #     browser.close()
    #
    # soup = BeautifulSoup(html, "html.parser")
    # print(soup)
    # titles = soup.find_all("h2", class_="card__title")
    #
    # for h2 in titles:
    #     print(h2.get_text(strip=True))
    # df.to_csv("created_data/repo_data.csv", index=False)
    return df


def scrape_repo_departments(current_time):
    base_url = "https://repository.ubn.ru.nl"
    base_dep = "https://repository.ubn.ru.nl/browse?type=authorganizationcode"
    # target_url_p2 = "https://repository.ubn.ru.nl/browse?rpp=50&sort_by=1&type=title&offset=50&etal=-1&order=ASC"


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

def load_or_create_data_repository(current_time, type_="departments"):
    # folder = {"departments": "department",
    #           "employees": "employees",
    #           "osiris": "osiris"}

    doc_name = {"departments": "department",
              "employees": "emp",
              "osiris": "osiris"}

    dir_name = f"created_data/repository"
    files = glob.glob(f"{dir_name}/{doc_name[type_]}*.csv")

    if files:
        latest = max(files, key=os.path.getmtime)
        ts = os.path.basename(latest).split("_")[1].split(".")[0]  # YYYYMMDD
        file_date = datetime.strptime(ts[:8], "%Y%m%d")

        ts = latest[latest.index("2"):latest.index(".")]
        if (datetime.now() - file_date).days <= 7:
            return pd.read_csv(latest), ts


    # Create new file
    if type_ == "departments":
        return scrape_repo_departments(current_time), current_time
    elif type_ == "employees":
        return scrape_full_repo(current_time), current_time

def scrape_full_repo(current_time):
    repo_departments, current_time = load_or_create_data_repository(current_time)
    emp_file_name = "created_data/repository/emp_" + current_time + ".csv"

    for index, row in repo_departments.iterrows():
        print(row["department"])
        print(row["url"])
        print("ind:", index)

        scrape_repo(row["url"], row["department"], emp_file_name)

    df = pd.read_csv(emp_file_name)
    return df


def scrape_employees():
    base_url = "https://www.ru.nl"
    target = "https://www.ru.nl/zoeken/scope/medewerkers?w="


    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/128.0.0.0 Safari/537.36",
        "Referer": target,
    }

    # PARAMS = {
    #     "rpp": 50,
    #     "sort_by": -1,
    #     "type": "authorganizationcode",
    #     "etal": -1,
    #     "order": "ASC"
    # }


    # create session from OG webpage
    html = get_html(target)
    soup = BeautifulSoup(html, "html.parser")

    # print(soup.prettify())
    # quit()
    TOTAL_RESULTS = 0

    TOTAL_RESULTS = soup.find(string=re.compile(r"\bResultaat\b", re.I))
    print("TOTAL_RESULTS", TOTAL_RESULTS)
    match_ = re.findall(r"(\d+)", TOTAL_RESULTS.string)
    TOTAL_RESULTS = int(match_[-1])
    RPP = int(match_[-2])
    print("RPP:", RPP)

    # quit()
    file_name = "created_data/employees/employees" + current_time + ".csv"

    file_headers = ["Name", "Url", "Faculties", "Keywords", "Onderzoeksthema", "Onderzoeksgroep", "Publicaties",
     "Onderzoeksbeurzen en -prijzen", "Projecten", "Onderwijs", "In de media", "Curriculum Vitae",
     "Nevenwerkzaamheden"]

    page_count = 0

    with open(file_name, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, file_headers )
        writer.writeheader()

        for offset in range(0, TOTAL_RESULTS, RPP):
            print("Offset", offset)

            employee_list = soup.find_all("h2", class_="card__title")
            # print("Employees:", employee_list)

            for emp in employee_list:
                print("------------------------------------------------")

                # get name
                name = emp.find("a")
                try:
                    emp_url = urljoin(base_url, name["href"])
                except:
                    continue


                print("URL: ", emp_url)

                name = emp.find("span", class_="link__text").string
                print("Name", name)

                data_dict = exctract_employee_page(emp_url, file_headers)
                data_dict["Name"] = name
                data_dict["Url"] = emp_url

                print("Dictionary keys:", data_dict.keys())


                for k in data_dict.keys():
                    if k not in file_headers:
                        print(k)
                        break
                writer.writerow(data_dict)

            print("------------------------------------------------")

            # next page
            page_count += 1
            target = f"https://www.ru.nl/zoeken/scope/medewerkers?w=&page={page_count}"

            html = get_html(target)
            soup = BeautifulSoup(html, "html.parser")
            print("Next page: succes!")

            #     break
            # break


                # faculty = emp.find_all("div", class_="meta-data")
                # print("Faculty", faculty)
                #
                #
                # faculty_names = faculty.find_all("a")
                # print("faculty names", faculty_names)

    print("Finished")
    return

def exctract_employee_page(url, file_headers):
    # url = "https://www.ru.nl/personen/kwisthout-j"

    html = get_html(url)
    page_soup = BeautifulSoup(html, "html.parser")

    # print(page_soup.prettify())

    """
    ["Name", "Faculteit", "Keywords", "url", "Onderzoeksthema", "Onderzoeksgroep", "Publicaties",
     "Onderzoeksbeurzen en -prijzen", "Projecten", "Onderwijs", "In de media", "Curriculum Vitae",
     "Nevenwerkzaamheden"] )
    """

    data_dict = {}
    # faculty/affiliations names
    faculty_names = page_soup.find("p", class_="text text--intro")

    if faculty_names:

        data_dict["Faculties"] = list(faculty_names.stripped_strings)
    else:
        data_dict["Faculties"] = "None found"

    print("Faculty names:", data_dict["Faculties"])

    # small_header_soup = page_soup.find_all("div", class_="profile__content")

    # onderzoeksthemas
    small_headers = page_soup.find_all("span", class_="label")

    # print("Small headers: ", small_headers)


    if small_headers:
        for h in small_headers:
            h_str = h.string
            # print("string: ", h_str)

            if h_str in file_headers:
                # print("header: ", h.find_next("ul", class_="list"))
                print("--------------")
                print("Header:", h_str)
                link_list = h.find_next("ul", class_="list")
                links = link_list.find_all("a")

                themas = []
                if links:
                    for link in links:
                        link_str = link.string
                        themas.append((link_str, link['href']))
                        print(link_str)
                # print("list:", themas)
                #     print(h_str)

                    data_dict[h_str] = themas


    keywords = page_soup.find_all("span", class_="meta-data__item")
    if keywords:
        kw_list = []
        for kw in keywords:
            kw_str = kw.string
            kw_list.append(kw_str)
        data_dict["Keywords"] = kw_list
        print("Keywords:", data_dict["Keywords"])
    big_headers = page_soup.find_all("h3", class_=["accordion-item"])
    # print("Big headers: ", big_headers)

    if big_headers:
        for h in big_headers:
            h_str = h.get_text(strip=True)
            print("--------------")
            print("Big Header:", h_str)

            if h_str in file_headers:
                link_list = h.find_next("ul", class_="list")
                if link_list:
                    papers = link_list.find_all("li")  # "span"
                    # urls = link_list.find_all("a")

                    themas = []
                    if papers:
                        for pap in papers:
                            url = pap.find_next("a")
                            tuptup = (pap.get_text(strip=True), url["href"])
                            # print("Tuple: ", tuptup)
                            themas.append(tuptup)
                            print(tuptup)

                        data_dict[h_str] = themas
            # break

    # print("Dictionary big: ", data_dict)
    # print("Dictionary keys: ", data_dict.keys())

    return data_dict


def get_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        html = page.content()
        browser.close()
    return html

def load_all_data():
    osiris_data = load_osiris_data("raw_data.xlsx")
    repo_data = pd.read_csv("created_data/repository/emp_20251115_161743.csv")
    employee_data = pd.read_csv("created_data/employees/employees20251115_161743.csv")
    return osiris_data, repo_data, employee_data

if __name__ == '__main__':
    # scrape()
    osiris_data = load_osiris_data("raw_data.xlsx")

    print("Data is loaded!")
    # print(data.iloc[0]['DOCENT_ROL'])

    # teachers = get_column_list(osiris_data, 'DOCENT_ROL')
    # print(teachers)

    # alphabet = string.ascii_lowercase

    # scrape_teachers()

    # scrape_repo()

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    repo_data, current_time = load_or_create_data_repository(current_time, "employees")

    print(repo_data.head())
    print("-----------------------------------------")

    # scrape_employees()
    employee_data = pd.read_csv("created_data/employees/employees20251115_161743.csv")

    print(employee_data.head())
    print("------------------------------------------")
    print("Data is loaded!")

    print("Employee data columns: ", employee_data.columns.to_list)
    print("Repository data columns: ", repo_data.columns.to_list)
    print("Osiris data columns: ", osiris_data.columns.to_list())







    """
    If (latest) dep_url files are older than 1 day, then use scrape_repo_departments()
    
    select latest dep_url file, use those data (dep, link) to access all papers etc.
    use scrape_repo()
    
    --------------------------------------------------
    
    create convo loop
    What topics do you want to find?
    
    
    
    """



# ----------------------------------------------------------------------------
# def scrape():
#     url = 'https://ru.osiris-student.nl/onderwijscatalogus/extern/cursussen'
#     url = "https://ru.osiris-student.nl/assets/i18n/nl.json"
#     url = "https://ru.osiris-student.nl/assets/config.json"
#     url = "https://rontw.osiris-student.nl"
#     url = "https://ru.osiris-student.nl/student/osiris/owc/zoekcriteria/cursus/owc_extern?limit=9999"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     print(soup)
#
#     # title = soup.select_one('title').text
#     # text = soup.select_one('p').text
#     # link = soup.select_one('a').get('href')
#
#     # print("title", title)
#     # print(text)
#     # print(link)
#
#
# if __name__ == '__main__':
#     # scrape()
#
#     # url = "https://ru.osiris-student.nl/assets/i18n/nl.json"
#     # url = "https://ru.osiris-student.nl/assets/config.json"
#     url = "https://rontw.osiris-student.nl"
#     # url = "https://ru.osiris-student.nl/student/osiris/owc/zoekcriteria/cursus/owc_extern?limit=9999"
#     params = {"page": 1, "limit": 100}
#     headers = {"User-Agent": "Mozilla/5.0"}
#
#     resp = requests.get(url, params=params, headers=headers)
#     data = resp.json()
#
#     print(data)
#
#     # df = pd.DataFrame(data["courses"])
#     # print(df.head())
