import sqlite3
import pandas as pd
import re

from rapidfuzz import fuzz

DATABASE = "search.db"

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # urls verwijderen
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # link, link1, link2 verwijderen
    text = re.sub(
        r"\blink\d*\b",
        " ",
        text
    )

    # extern verwijderen
    text = re.sub(
        r"\bextern\b",
        " ",
        text
    )

    # speciale tekens verwijderen
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # dubbele spaties
    text = " ".join(text.split())

    return text


def match_course(course_name, education_text):
    course_name = clean_text(course_name)

    education_text = clean_text(education_text)

    if not course_name:
        return 0, "none"

    if course_name in education_text:
        return 100, "exact"

    token_score = fuzz.token_set_ratio(course_name, education_text)

    partial_score = fuzz.partial_ratio(course_name, education_text)

    score = max(token_score, partial_score)

    if score >= 85:
        return score, "fuzzy"

    return score, "none"


conn = sqlite3.connect(DATABASE)

employees = pd.read_sql(
    """
    SELECT
        name,
        url,
        onderwijs
    FROM employee
    """,
    conn
)

courses = pd.read_sql(
    """
    SELECT
        cursus,
        lange_naam
    FROM osiris
    """,
    conn
)

print("Employees:", len(employees))

print("Courses:", len(courses))

# dubbele vaknamen verwijderen

courses = courses.drop_duplicates(subset=["cursus"])

print("Unique courses:", len(courses))

matches = []

exact_matches = 0
fuzzy_matches = 0

print("matching: ")

for _, employee in employees.iterrows():

    education = employee["onderwijs"]

    if pd.isna(education):
        continue

    for _, course in courses.iterrows():

        score, match_type = match_course(course["lange_naam"], education)

        if match_type != "none":

            matches.append(
                (
                    course["cursus"],
                    employee["name"],
                    employee["url"],
                    int(score)
                )
            )

            if match_type == "exact":
                exact_matches += 1

            elif match_type == "fuzzy":
                fuzzy_matches += 1

print()
print("Matches gevonden:", len(matches))

print("Exact:", exact_matches)

print("Fuzzy:", fuzzy_matches)

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
    """
)

cur.executemany(
    """
    INSERT INTO course_employee
    VALUES (?,?,?,?)
    """,
    matches
)

conn.commit()

# checkas

count = pd.read_sql(
    """
    SELECT COUNT(*)
    FROM course_employee
    """,
    conn
)

print()
print(count)

examples = pd.read_sql(
    """
    SELECT *
    FROM course_employee
    LIMIT 10
    """,
    conn
)

print(examples)
conn.close()

print()
print("Done!")
