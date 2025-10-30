# # importing libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import string

def load_data(filename):
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


def scrape_teachers(last_name, initial):
    #https://realpython.com/beautiful-soup-web-scraper-python/

    # Radboud web zoekfunctie
    # https://www.ru.nl/zoeken?w=
    # https://www.ru.nl/zoeken/scope/medewerkers?w=

    last_name, initial = last_name.lower().strip(), initial.lower().strip()
    # print(f'Name: {last_name}, {initial}')

    url = f"https://www.ru.nl/personen/{last_name}-{initial}"
    # print(f'URL: {url}')
    # url = f"https://www.ru.nl/personen/brasjen-m"

    url =f"https://www.ru.nl/personen/kwisthout-j"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print("soup")

    # TODO
    # Collect every small header (automatically)
    headers = ['Onderzoeksthema', 'Onderzoeksgroep']

    # Collect every dropdown menu with content (automatically)

    expertise_header = soup.find("span", class_="label", string="Onderzoeksthema")

    container = expertise_header.find_next_sibling()
    all_text = ";".join(container.stripped_strings)
    print(f' Name: {last_name}, {initial}')
    print(all_text)

    return


if __name__ == '__main__':
    # scrape()
    data = load_data("raw_data.xlsx")

    print("Data is loaded!")
    # print(data.iloc[0]['DOCENT_ROL'])

    teachers = get_column_list(data, 'DOCENT_ROL')
    # print(teachers)

    alphabet = string.ascii_lowercase


    scrape_teachers("","")
    # for i in range(3):
    #     for l in alphabet:
    #         try:
    #             scrape_teachers(teachers[i], l)
    #             break
    #         except Exception as e:
    #             continue






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
