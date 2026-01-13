from functools import lru_cache

import pandas as pd
import re
import spacy
from scraper import load_all_data

# Load small NLP model (tokenization, lemmatization)
# spacy.load('nl_core_news_lg')
nlp = spacy.load("nl_core_news_lg")

# osiris_data, repo_data, employee_data = load_all_data()

repo_data = pd.read_csv("created_data/repository/emp_20251115_161743.csv")
employee_data = pd.read_csv("created_data/employees/employees20251115_161743.csv")
osiris_data = pd.read_excel("created_data/CORRECT_DATA/raw_data.xlsx")

@lru_cache(maxsize=50000)
def preprocess_cached(text: str):
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_punct and not t.is_space]
    return " ".join(tokens)

def preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return preprocess_cached(text)
    # return preprocess_cached(text)

def count_occurrences(text, keyword):
    processed = preprocess(text)
    key = preprocess(keyword)
    return len(re.findall(rf"{re.escape(key)}", processed))

def compute_scores(keyword, employee_data, repo_data, osiris_data):
    scores = {}

    # Employee Data
    print("Employee data")
    for _, row in employee_data.iterrows():
        name = row.get("Name", "Unknown")
        # print("Name:", name)

        total = 0
        for col in ["Keywords", "Publicaties", "Onderwijs", "In de media", "Projecten"]:
            if col in employee_data.columns:
                value = row.get(col, "")
                if isinstance(value, list):
                    # Join list items into a single string
                    value = " ".join(str(v) for v in value)
                # print("Value:", value)
                total += count_occurrences(value, keyword)

        # print("Total:", total)
        if total > 0:
            scores[name] = scores.get(name, 0) + total


    print("-------------------")
    print("Repo data")
    # Repo Data
    for _, row in repo_data.iterrows():
        authors = row.get("authors", [])

        if isinstance(authors, str):
            authors = [authors]
        print(f"authors: {authors}")
        total = sum(count_occurrences(row.get(col, ""), keyword)
                    for col in ["title", "keywords", "publishing_info"]
                    if col in repo_data.columns)

        if total > 0:
            for author in authors:
                scores[author] = scores.get(author, 0) + total

    print("------------------")
    print("Osiris data")
    # Osiris Data
    for _, row in osiris_data.iterrows():
        instructor = row.get("DOCENT_ROL", "Unknown")
        print(f"Instructor: {instructor}")
        total = sum(count_occurrences(row.get(col, ""), keyword)
                    for col in ["Aims", "LANGE_NAAM_NL"]  # "INHOUD",
                    if col in osiris_data.columns)

        if total > 0:
            scores[instructor] = scores.get(instructor, 0) + total

    print("------------------")
    print("scores done")

    return scores

def search_keyword_loop(employee_data, repo_data, osiris_data):
    # print("Type 'quit' to exit the keyword search.")

    while True:
        print("------------------------------------------")
        print("Type 'quit' to exit the keyword search.")
        # keyword = input("Enter keyword to search: ").strip()
        keyword = "ethiek"
        if keyword.lower() == "quit":
            print("Exiting search loop.")
            break

        try:
            # max_results = int(input("Maximum results: "))
            max_results = 20
        except ValueError:
            print("Invalid number. Try again. ")
            continue

        scores = compute_scores(keyword, employee_data, repo_data, osiris_data)

        if not scores:
            print(f"No results for '{keyword}'.")
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print(f"Top {max_results} results for keyword '{keyword}':")

        for name, score in ranked[:max_results]:
            print(f"{name}: {score}\n")

        # print()

search_keyword_loop(employee_data, repo_data, osiris_data)

