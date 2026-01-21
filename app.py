# ---------------------------------------------------

import base64

import pandas as pd
import re
import spacy
import streamlit as st
from functools import lru_cache
from scraper import load_all_data
from deep_translator import GoogleTranslator
from langdetect import detect

# Load small NLP model (tokenization, lemmatization)
nlp = spacy.load("nl_core_news_lg")

osiris_data, repo_data, employee_data = load_all_data()

# @lru_cache(maxsize=50000)
def preprocess_cached(text: str):
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_punct and not t.is_space]
    return " ".join(tokens)

def preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return preprocess_cached(text)

def count_occurrences(text, keyword):
    processed = preprocess(text)
    key = preprocess(keyword)
    return len(re.findall(rf"{re.escape(key)}", processed))


def compute_scores(keyword, employee_data, repo_data, osiris_data):
    scores = {}

    # Employee Data
    for _, row in employee_data.iterrows():
        name = row.get("Name", "Unknown")

        total = 0
        for col in ["Keywords", "Publicaties", "Onderwijs", "In de media", "Projecten"]:
            if col in employee_data.columns:
                value = row.get(col, "")
                if isinstance(value, list):
                    # Join list items into a single string
                    value = " ".join(str(v) for v in value)
                total += count_occurrences(value, keyword)

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
                    for col in ["INHOUD", "Aims", "LANGE_NAAM_NL"]
                    if col in osiris_data.columns)

        if total > 0:
            scores[instructor] = scores.get(instructor, 0) + total

    return scores


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


def run_streamlit_ui(employee_data, repo_data, osiris_data):
    st.set_page_config(page_title="", layout="wide")
    # st.image('app_layouts/RU_LOGO_COMPLEET.png', width=300)
    # st.logo('app_layouts/RU_LOGO_COMPLEET.png', link="https://www.ru.nl/", size="large")

    # Create three columns, with the middle one taking most space
    # col1, col2, col3 = st.columns([1, 3, 1])  # Relative widths
    #
    # # Place the image in the middle column
    # with col2:
    #     st.image("app_layouts/RU_LOGO_COMPLEET.png",)

    # Banner with centered logo
    # st.markdown(
    #     """
    #     <style>
    #     .top-banner {
    #         width: 100%;
    #         background-color: #f5f5f5;
    #         padding: 15px 0;
    #         text-align: center;
    #         border-bottom: 1px solid #ddd;
    #     }
    #     .top-banner img {
    #         height: 60px;
    #     }
    #     </style>
    #
    #     <div class="top-banner">
    #         <img src="app_layouts/RU_LOGO_COMPLEET.png">
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    logo_base64 = get_base64_image("app_layouts/RU_LOGO_COMPLEET.png")

    st.markdown(
        f"""
        <style>
        .top-banner {{
            width: 100%;
            background-color: #ffffff;
            padding: 15px 0;
            text-align: center;
            border-bottom: 0px solid #ddd;
        }}
        .top-banner img {{
            height: 150px;
        }}
        </style>

        <div class="top-banner">
            <a href="https://www.ru.nl" target="_blank">
            <img src="data:image/png;base64,{logo_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Custom CSS for centering the image and adjusting spacing
    # st.markdown("""
    # <style>
    # .big-logo {
    #     display: block;
    #     margin-left: auto;
    #     margin-right: auto;
    #     width: 50%; /* Adjust width as needed */
    #     max-width: 400px; /* Prevents it from getting too big */
    # }
    # /* Adjust main content padding to bring it closer to the top */
    # .stApp > div[data-testid="stAppViewContainer"] > div {
    #     padding-top: 2rem;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    #
    # # Display the logo
    # st.markdown('<img src="app_layouts/RU_LOGO_COMPLEET.png" class="big-logo">', unsafe_allow_html=True)

    st.title("Keyword Search Across Databases")
    st.write("EN: This tool is the first version of a tool created for Radboud Universiteit - Onderwijs voor Professionals (OvP).\n"
             "It works best when using singular keywords such as 'ethics' or 'artificial intelligence'.\n"
             "Try to avoid full phrases like 'I\'m looking for ...'")

    st.write(
        "NL: Deze tool is de eerste versie van een ontwikkeling gedaan voor Radboud Universiteit - Onderwijs voor Professionals (OvP).\n"
        "Het werkt het beste op simpele trefwoorden zoals 'ethiek' of 'artificial intelligence'.\n"
        "Probeer zinnen zoals 'ik ben op zoek naar ...' te vermijden.")


    keyword = st.text_input("Enter keyword")

    translated = GoogleTranslator(source='auto', target='nl').translate(keyword)

    key_dict = simple_search(employee_data, repo_data, osiris_data, keyword)
    t_dict2 = simple_search(employee_data, repo_data, osiris_data, translated)

    max_results = st.number_input("Max results", min_value=5, value=5)

    if st.button("Search"):
        st.write("Search results:")
        st.write(key_dict)

        st.write("Search results:")
        st.write(t_dict2)
        # scores = compute_scores(keyword, employee_data, repo_data, osiris_data)
        #
        # if not scores:
        #     st.write("No results found.")
        #     return
        #
        # ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # st.subheader("Results")
        # for name, score in ranked[:max_results]:
        #     st.write(f"**{name}** — {score} occurrences")


run_streamlit_ui(employee_data, repo_data, osiris_data)


