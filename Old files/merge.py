import os
import re
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Text processing
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nlp = spacy.load("nl_core_news_lg")

repo_data = pd.read_csv("created_data/repository/emp_20251115_161743.csv")
employee_data = pd.read_csv("created_data/employees/employees20251115_161743.csv")
osiris_data = pd.read_excel("created_data/CORRECT_DATA/raw_data_inh_doel.xlsx")


def preprocess_data(data):
    emp_copy = data.copy()
    employee_cols = emp_copy.columns.tolist()
    employee_merged = []

    nlp = spacy.load("nl_core_news_lg")

    for i, row in emp_copy.iterrows():
        name = ""
        if 'Name' in employee_cols:
            name = row['Name']
        elif 'authors' in employee_cols:
            name = row['authors']
        elif 'DOCENT_ROL' in employee_cols:
            name = "error"
        text = ""

        for c in employee_cols:
            if c not in ["Name", "Url"] and not pd.isna(row[c]):
                # text += f"{row[c]}\n"
                text += " ".join(token.lemma_ for token in nlp(row[c]) if not token.is_stop and token.is_alpha)

        # print(text)
        employee_merged.append([name, text])
        # break

    employee_merged = pd.DataFrame(employee_merged, columns=["Name", "Text"])
    print(employee_merged.head())
    return employee_merged


def preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return text


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
            col = 'text'
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

    # Repo Data
    for _, row in repo_data.iterrows():
        authors = row.get("authors", [])

        if isinstance(authors, str):
            authors = [authors]

        total = sum(count_occurrences(row.get(col, ""), keyword)
                    for col in ["title", "keywords", "publishing_info"]
                    if col in repo_data.columns)

        if total > 0:
            for author in authors:
                scores[author] = scores.get(author, 0) + total

    # Osiris Data
    for _, row in osiris_data.iterrows():
        instructor = row.get("DOCENT_ROL", "Unknown")
        total = sum(count_occurrences(row.get(col, ""), keyword)
                    for col in ["Aims", "LANGE_NAAM_NL"]  # "INHOUD",
                    if col in osiris_data.columns)

        if total > 0:
            scores[instructor] = scores.get(instructor, 0) + total

    return scores

def search_keyword_loop(employee_data, repo_data, osiris_data):
    # print("Type 'quit' to exit the keyword search.")

    while True:
        print("------------------------------------------")
        print("Type 'quit' to exit the keyword search.")
        keyword = input("Enter keyword to search: ").strip()
        # keyword = "ethiek"
        if keyword.lower() == "quit":
            print("Exiting search loop.")
            break

        try:
            max_results = int(input("Maximum results: "))
            # max_results = 20
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

def combine_excel():
    data = pd.read_excel("created_data/CORRECT_DATA/raw_data_inh_doel.xlsx")
    print(data.columns)
    # Zet INHOUD en DOEL naast elkaar
    df_wide = data.pivot_table(
        index=["CURSUS", "LANGE_NAAM_NL", "DOCENT_ROL"],
        columns="LABEL",
        values="OSS_ADF_UTILITY.HTML_TO_TEXT(H.INHOUD)",
        aggfunc="first"
    ).reset_index()

    # Kolomnamen opschonen
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={
        "INHOUD": "INHOUD",
        "DOEL": "DOEL"
    })

    # Opslaan naar nieuw Excel-bestand
    df_wide.to_excel("created_data/CORRECT_DATA/combined.xlsx", index=False)

    print(df_wide.iloc[-1])
    print(df_wide.columns)
    print("succes!")
    return

# combine_excel()

def simple_search(employee_data, repo_data, osiris_data, keyword):
    final_dict = {}
    data = [employee_data, repo_data, osiris_data]
    # iterate over data
    for d in data:
        print("------------------------------------------")
        cols = d.columns.tolist()
        for i, row in d.iterrows():
            # print(f"{i}: {row}")
            count = 0
            name = ""
            for j, col in enumerate(cols):
                # print(f"{j}: {col}")
                if col in ["authors", "Name", "DOCENT_ROL"]:
                    name = row[col]
                    # print(f"name: {name}")
                    continue

                # print(f"Row: {row[col]}")
                if name in final_dict.keys():
                    count = final_dict[name]

                if pd.isna(row[col]):
                    continue

                if isinstance(row[col], list):
                    temp_text = " ".join(row[col])
                else:
                    temp_text = row[col]

                # print(f"type: {type(temp_text)}")

                if keyword in temp_text.lower():
                    count += temp_text.count(keyword)

            if count > 0:
                print(f"{name}: {count}\n")
                final_dict[name] = count
    return final_dict

final = simple_search(employee_data, repo_data, osiris_data, "ethiek")
print(final)

    # check if keyword appears, count appearance
    # store person: count in dict
    #
    # repeat for all data sources





# employee_data = preprocess_data(employee_data)
# print("Done")
# osiris_data = preprocess_data(osiris_data)
# print("Done")
# repo_data = preprocess_data(repo_data)
# print("Done")
# search_keyword_loop(employee_data, repo_data, osiris_data)
