import sqlite3
import pandas as pd
from utils import *
from rapidfuzz import process, fuzz

import re

def normalize_name(name):
    name = str(name).lower()

    # verwijder roepnaam tussen haakjes
    name = re.sub(r"\(.*?\)", "", name)

    # verwijder punten
    name = name.replace(".", "")

    # dubbele spaties weg
    name = " ".join(name.split())

    return name


def match_employee(docent):

    docent = normalize_name(docent)

    match = process.extractOne(
        docent,
        employee_lookup.keys(),
        scorer=fuzz.token_sort_ratio
    )

    if match is None:
        return None

    matched_name, score, _ = match

    if score < 90:
        return None

    return employee_lookup[matched_name]

def get_docent_cursussen(employee_name, O):
    mask = O["DOCENT_ROL"].apply(
        lambda x: docent_match(employee_name, x)
    )

    return O.loc[mask, ["CURSUS"]]


EMPLOYEE_PATH = "created_data/cleaned_data/employee.csv"
OSIRIS_PATH = "created_data/cleaned_data/osiris.csv"
DATABASE = "data/search.db"

E = pd.read_csv(EMPLOYEE_PATH)
O = pd.read_csv(OSIRIS_PATH)

employee_lookup = {}

for _, row in E.iterrows():

    normalized = normalize_name(row["Name"])

    employee_lookup[normalized] = {
        "name": row["Name"],
        "url": row["Url"]
    }



conn = sqlite3.connect(DATABASE)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS course_employee")

cur.execute("""
CREATE TABLE course_employee(

    course TEXT,

    employee TEXT,

    employee_url TEXT
)
)
""")

rows = []

for _, row in O.iterrows():

    cursus = row["CURSUS"]

    docent_rol = row["DOCENT_ROL"]
    docenten = docent_rol.split(",")

    for docent in docenten:

        employee = match_employee(docent)

        if employee is None:
            continue

    rows.append(

        (
            cursus,
            employee["name"],
            employee["url"]
        )

    )

cur.executemany(
    """
    INSERT INTO course_employee
    VALUES (?,?,?)
    """,
    rows
)

conn.commit()
conn.close()
# print("done")