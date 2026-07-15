import sqlite3
import pandas as pd
import re

from rapidfuzz import fuzz

EMPLOYEE_PATH = "created_data/cleaned_data/employee.csv"
OSIRIS_PATH = "created_data/cleaned_data/osiris.csv"

DATABASE = "search.db"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"http\S+",
        " ",
        text)

    text = re.sub(
        r"\blink\d+\b",
        " ",
        text)

    text = re.sub(
        r"\blink\b",
        " ",
        text)

    text = re.sub(
        r"\bextern\b",
        " ",
        text)

    # speciale tekens
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text)

    # dubbele spaties
    text = " ".join(text.split())

    return text


def clean_course_name(course):
    return clean_text(course)


def course_in_employee_text(course, employee_text):
    course = clean_course_name(course)

    employee_text = clean_text(employee_text)

    if not course:
        return 0

    # exacte match

    if course in employee_text:
        return 100

    # fuzzy fallback
    score = fuzz.partial_ratio(course, employee_text)

    return score


E = pd.read_csv(EMPLOYEE_PATH)
O = pd.read_csv(OSIRIS_PATH)

print("Employees:", len(E))
print("Courses:", len(O))

osiris_courses = []

for _, row in O.iterrows():

    course = row["CURSUS"]

    if pd.notna(course):
        osiris_courses.append(course)

# dubbele vakken verwijderen

osiris_courses = list(set(osiris_courses))

print("Unique Osiris courses:", len(osiris_courses))




matches = []

print("start matching: ")

for _, employee in E.iterrows():

    # medewerkers zonder onderwijs overslaan

    if pd.isna(employee["Onderwijs"]):
        continue

    education = employee["Onderwijs"]

    for course in osiris_courses:

        score = course_in_employee_text(course, education)

        if score >= 90:
            matches.append(
                (
                    course,
                    employee["Name"],
                    employee["Url"],
                    score
                )
            )

print("Matches gevonden:",  len(matches))
print("DB: ")

conn = sqlite3.connect(DATABASE)
cur = conn.cursor()

cur.execute(
    """
    DROP TABLE IF EXISTS course_employee
    """
)

cur.execute(
    """
    CREATE TABLE course_employee(
        course TEXT,
        employee TEXT,
        employee_url TEXT,
        match_score INTEGER
    )
    """)

cur.executemany(
    """
    INSERT INTO course_employee
    VALUES (?,?,?,?)
    """,
     matches)

conn.commit()
conn.close()

print("Done!")
