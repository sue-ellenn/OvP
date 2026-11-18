
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
import base64

import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from io import BytesIO
import requests
import sys
import os
import openai
import uuid

from feedback import init_feedback_db, save_employee_feedback
if "feedback_session_id" not in st.session_state:
    st.session_state.feedback_session_id = str(uuid.uuid4())

if "given_feedback" not in st.session_state:
    st.session_state.given_feedback = set()

# os.chdir('..')
# sys.path.insert(1, os.getcwd())
sys.path.append(str(Path(__file__).resolve().parent.parent))

print("PATH: ", sys.path)
from database import *
from utils import *
from search import *
from llm_client import answer_question
import inspect

print("LLM CLIENT:", inspect.getfile(answer_question))
print("SIGNATURE:", inspect.signature(answer_question))
# os.chdir('pages')
# from build_course_mapping import *
from urllib.parse import quote

api_key = st.secrets["LLM_API_KEY"]
base_url = st.secrets["LLM_BASE_URL"]
llm_model = st.secrets["LLM_MODEL"]

# pd.read_csv("created_data/cleaned_data/repo.csv").to_parquet('created_data/cleaned_data/repo.parquet', compression="snappy")

FILES = {
    "search.db": "https://github.com/sue-ellenn/OvP/releases/download/data/search.db",
    "embeddings.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy",
    "meta.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/meta.npy"
}
# https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy

# DATA_DIR = Path("data")
# DATA_DIR.mkdir(exist_ok=True)
#
# DATA_DIR = Path("data")
# DATA_DIR.mkdir(exist_ok=True)
# DB_PATH = "search.db"
# # EMBEDDINGS_PATH = FILES["embeddings.npy"]
# EMBEDDINGS_PATH = "data/embeddings.npy"
#
# response = requests.get(EMBEDDINGS_PATH)
# response.raise_for_status()
#
# embeddings = np.load(BytesIO(response.content))
# embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

TOP_FTS = 30
TOP_FINAL = 10

embeddings, meta, model = load_resources()
conn = get_connection()


#
# def create_linkedin_url(query, location=None, company=None, title=None):
#     params = []
#
#     if query:
#         params.append(f"keywords={quote(query)}")
#     if location:
#         params.append(f"geoUrn={quote(location)}")
#     if company:
#         params.append(f"currentCompany={quote(company)}")
#     if title:
#         params.append(f"title={quote(title)}")
#
#     return "https://www.linkedin.com/search/results/people/?" + "&".join(params)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


def create_linkedin_url(query):
    return f"https://www.linkedin.com/search/results/people/?keywords={quote(query)}"


def render_single_result(result, conn, show_feedback=False, query=None):
    name = result["name"]
    source = result["source"]
    explanation = result.get("explanation", "")

    # Medewerkers
    if source == "Employees":
        employee = get_employee(name, conn)

        if employee.empty:
            st.markdown("_Geen medewerker gevonden._")
            return

        employee = employee.iloc[0]

        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(f"### {employee['name']}")

            if pd.notna(employee["url"]):
                st.markdown(f"[Medewerkerspagina]({employee['url']})")

            if pd.notna(employee["faculties"]):
                st.markdown(f"**Faculteit:** {employee['faculties']}")

            if pd.notna(employee["onderzoeksthema"]):
                st.markdown(
                    f"**Onderzoeksthema:** {employee['onderzoeksthema']}"
                )

            if pd.notna(employee["onderzoeksgroep"]):
                st.markdown(
                    f"**Onderzoeksgroep:** {employee['onderzoeksgroep']}"
                )

            if explanation:
                st.markdown("**Waarom relevant?**")
                st.write(explanation)

        with col2:
            if show_feedback:
                feedback_key = f"{employee['name']}_{query}"

                if "given_feedback" not in st.session_state:
                    st.session_state.given_feedback = set()

                if feedback_key in st.session_state.given_feedback:
                    st.success("👍")
                elif st.button(
                    "👍",
                    key=f"feedback_{feedback_key}",
                    help="Deze medewerker was relevant"
                ):
                    save_employee_feedback(
                        name=employee["name"],
                        url=(
                            employee["url"]
                            if pd.notna(employee["url"])
                            else ""
                        ),
                        query=query,
                        session_id=st.session_state.feedback_session_id
                    )

                    st.session_state.given_feedback.add(feedback_key)
                    st.rerun()

        courses = get_courses_for_employee(employee["name"], conn)

        if not courses.empty:
            with st.expander("Geeft onderwijs in"):
                for _, course in courses.iterrows():
                    st.write(f"- {course['course']}")

    # Cursussen
    elif source == "Osiris":
        course = get_osiris_course(name, conn)

        if course is None:
            st.markdown("_Geen cursusdetails gevonden._")
            return

        st.markdown(
            f"### {course['lange_naam']} ({course['cursus']})"
        )

        if explanation:
            st.markdown("**Waarom relevant?**")
            st.write(explanation)

        employees = get_employees_for_course(course["cursus"], conn)

        if not employees.empty:
            with st.expander("Docenten"):
                for _, employee in employees.iterrows():
                    st.markdown(f"**{employee['employee']}**")

                    if pd.notna(employee["employee_url"]):
                        st.markdown(
                            f"[Medewerkerspagina]({employee['employee_url']})"
                        )

        with st.expander("Vak informatie"):
            if pd.notna(course["inhoud"]):
                st.markdown(f"**Inhoud:** {course['inhoud']}")

            if pd.notna(course["doel"]):
                st.markdown(f"**Doel:** {course['doel']}")

    # Publicaties
    elif source == "Repo":
        paper = get_repository_record(name, conn)

        if paper is None:
            st.markdown("_Geen publicatiegegevens gevonden._")
            return

        st.markdown(f"### {paper['title']}")

        if explanation:
            st.markdown("**Waarom relevant?**")
            st.write(explanation)

        if pd.notna(paper["authors"]):
            st.markdown(f"**Auteurs:** {paper['authors']}")

        if pd.notna(paper["department"]):
            st.markdown(f"**Afdeling:** {paper['department']}")

        if pd.notna(paper["keywords"]):
            st.markdown(f"**Keywords:** {paper['keywords']}")

        if pd.notna(paper["title_url"]):
            st.markdown(
                f"[Publicatie bekijken]({paper['title_url']})"
            )

    else:
        st.warning(f"Onbekende bron: {source}")

# ----------------------------
# UI building
st.set_page_config(page_title="Expert Finder", page_icon="🔎", layout="wide")


# if "selected_terms" not in st.session_state:
#     st.session_state.selected_terms = []

# st.sidebar.header("Plotting Demo")

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

st.title("Onderwijs voor Professionals: Expert finder")
# st.info(
#     "EN: This tool is the first version of a tool created for Radboud Universiteit - Onderwijs voor Professionals (OvP).\n"
#     "It works best when using singular keywords such as 'ethics' or 'artificial intelligence'.\n"
#     "Try to avoid full phrases like 'I\'m looking for ...'\n\n"
#     "NL: Deze tool is de eerste versie van een ontwikkeling gedaan voor Radboud Universiteit - Onderwijs voor Professionals (OvP).\n"
#     "Het werkt het beste op simpele trefwoorden zoals 'ethiek' of 'artificial intelligence'.\n"
#     "Probeer zinnen zoals 'ik ben op zoek naar ...' te vermijden.", icon="⚠️"
# )

st.info("""
### Welkom bij de Expert Finder

Met de **Expert Finder** kun je eenvoudig zoeken naar kennis en expertise binnen de Radboud Universiteit. De AI-assistent doorzoekt medewerkersprofielen, cursusinformatie en publicaties om relevante resultaten te vinden.

**Hoe gebruik je de Expert Finder?**
1. Stel een vraag over een onderwerp of expertise waar je naar op zoek bent.
2. Bekijk de gevonden medewerkers, cursussen en publicaties.
3. Stel eventueel een vervolgvraag om je zoekopdracht te verfijnen.

**Voorbeeldvraag:**

*Welke medewerkers houden zich bezig met artificial intelligence en machine learning binnen de Radboud Universiteit?*

Vergeet niet om relevante resultaten een duimpje te geven! Daar wordt het systeem steeds beter van :)

**Tip:** Je hoeft geen specifieke zoektermen te kennen. Stel je vraag gewoon in je eigen woorden. Hoe concreter je vraag, hoe gerichter de resultaten.

Veel succes met ontdekken!
""", icon="🔎")

# st.warning(
#         "NL: Deze tool is de eerste versie van een ontwikkeling gedaan voor Radboud Universiteit - Onderwijs voor Professionals (OvP).\n"
#         "Het werkt het beste op simpele trefwoorden zoals 'ethiek' of 'artificial intelligence'.\n"
#         "Probeer zinnen zoals 'ik ben op zoek naar ...' te vermijden.")

# pass


# AI-assistent voor Expert finder

# st.subheader("AI-assistent")



# ---------------------------------------------------------
# Chat initialiseren
# ---------------------------------------------------------

if "ai_answer" not in st.session_state:
    st.session_state.ai_answer = None

if "ai_query" not in st.session_state:
    st.session_state.ai_query = ""

if "ai_history" not in st.session_state:
    st.session_state.ai_history = []

if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

if "feedback_session_id" not in st.session_state:
    st.session_state.feedback_session_id = str(uuid.uuid4())

if "given_feedback" not in st.session_state:
    st.session_state.given_feedback = set()

init_feedback_db()


# ---------------------------------------------------------
# Eerdere chatberichten weergeven
# ---------------------------------------------------------

for message in st.session_state.ai_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Nieuwe vraag verwerken
# ---------------------------------------------------------

query = st.chat_input("Waar ben je naar op zoek?")

if query:
    # Bestaande geschiedenis meegeven, zonder de nieuwe vraag
    # hier alvast dubbel aan toe te voegen.
    history = st.session_state.ai_history.copy()

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner(
            "Relevante medewerkers, cursussen en publicaties zoeken..."
        ):
            try:
                answer, results = answer_question(
                    query,
                    history,
                    conn
                )

                # Antwoord controleren
                if not isinstance(answer, dict):
                    answer = {
                        "introduction": "Ik heb relevante resultaten gevonden.",
                        "results": [],
                        "conclusion": ""
                    }

                if not isinstance(results, pd.DataFrame):
                    results = pd.DataFrame()

                if "explanation" not in results.columns:
                    results["explanation"] = ""

                # Volledig antwoord opslaan
                st.session_state.ai_answer = answer
                st.session_state.ai_results = results
                st.session_state.ai_query = query

                # Gesprekscontext opbouwen
                history_text = answer.get("introduction", "")

                if not results.empty:
                    history_text += "\n\nEerder geselecteerde resultaten:\n"

                    for _, row in results.iterrows():
                        history_text += (
                            f"\n- Naam: {row['name']}"
                            f"\n  Bron: {row['source']}"
                            f"\n  Relevantie: {row.get('explanation', '')}\n"
                        )

                history_text += "\n" + answer.get("conclusion", "")

                # Gebruikersvraag en AI-antwoord ieder één keer opslaan
                st.session_state.ai_history.append({
                    "role": "user",
                    "content": query
                })

                st.session_state.ai_history.append({
                    "role": "assistant",
                    "content": history_text
                })

            except Exception as e:
                st.error(f"De AI-assistent kon geen antwoord geven: {e}")
                st.exception(e)



if st.session_state.ai_results is not None:

    answer = st.session_state.ai_answer or {}
    results = st.session_state.ai_results
    query = st.session_state.ai_query

    introduction = answer.get(
        "introduction",
        "Ik heb relevante resultaten gevonden."
    )

    conclusion = answer.get("conclusion", "")

    # Introductie altijd weergeven
    if introduction:
        st.markdown(introduction)

    # Resultaatkaarten
    if not results.empty:
        st.divider()

        for _, result in results.iterrows():
            with st.container(border=True):
                render_single_result(
                    result,
                    conn,
                    show_feedback=(result["source"] == "Employees"),
                    query=query
                )

        # Conclusie altijd weergeven, ook als er geen kaarten zijn
        if conclusion:
            st.divider()
            st.markdown(conclusion)

    else:
        st.info(
            "Ik heb geen relevante medewerkers, cursussen of "
            "publicaties kunnen vinden."
        )

        if conclusion:
            st.markdown(conclusion)

    # Onderliggende zoekresultaten
    with st.expander("Bekijk de onderliggende zoekresultaten"):
        if not results.empty:
            st.dataframe(
                results.drop(
                    columns=["explanation"],
                    errors="ignore"
                ),
                width="stretch"
            )
        else:
            st.write("Geen zoekresultaten.")


# query = st.text_input("Zoekterm(en)/Search term(s)")
# TOP_FINAL = st.number_input("Max resultaten/results", min_value=1, max_value=150, value=10)

# with st.chat_message("assistant"):
#     st.write("Hello human")

# -----------------------------------
# FIXME
# @st.dialog("Mijn Pop-up Venster")
# def toon_popup():
#     st.write("Dit is de inhoud van het pop-upvenster.")
#     if st.button("Sluiten"):
#         st.rerun()


#
# if st.button("Open pop-up"):
#     toon_popup()
#
# # Pop_over is beter
# # Add pop_over per professional met al hun info
# # TODO FIXME
# with st.popover("Open pop-over"):
#     st.write("Hallo vanuit de pop-over!")
#     naam = st.text_input("Wat is je naam?")

# -----------------------------------

selected_terms = []
