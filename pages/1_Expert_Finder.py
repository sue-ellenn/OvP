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


# print("all columns: ", )
#
# print(pd.read_sql("PRAGMA table_info(employee)", conn))
# print(pd.read_sql("PRAGMA table_info(osiris)", conn))
# print(pd.read_sql("PRAGMA table_info(repo)", conn))
# print(pd.read_sql("PRAGMA table_info(course_employee)", conn))

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

def create_linkedin_url(query):
    return f"https://www.linkedin.com/search/results/people/?keywords={quote(query)}"


def render_single_result(result, conn):
    """
    generates a single result in the required format
    @param result: search result
    @param conn: sqlit3 connection
    @return:
    """
    name = result["name"]
    source = result["source"]

    # if source == "Osiris":
    #     course = get_osiris_course(name, O)
    #     name = course['LANGE_NAAM_NL']

    # st.markdown("### " + name)
    # st.caption(f"Bron: {source}")

    # employees
    if source == "Employees":
        employee = get_employee(name, conn)
        if employee.empty:
            st.markdown("_Geen medewerker gevonden._")
            return

        employee = employee.iloc[0]

        st.markdown(f"## {employee['name']}")

        if pd.notna(employee["url"]):
            st.markdown(f"[Medewerkerspagina]({employee['url']})")

        if pd.notna(employee["faculties"]):
            st.markdown(f"**Faculteit:** {employee['faculties']}")

        if pd.notna(employee["onderzoeksthema"]):
            st.markdown(f"**Onderzoeksthema:** {employee['onderzoeksthema']}")

        if pd.notna(employee["onderzoeksgroep"]):
            st.markdown(f"**Onderzoeksgroep:** {employee['onderzoeksgroep']}")

        courses = get_courses_for_employee(employee["name"], conn)

        if not courses.empty:
            st.markdown("### Geeft onderwijs in:")

            with st.expander("Onderwijs"):
                for _, course in courses.iterrows():
                    # add course name
                    st.write(f"- {course['course']}: {None}")


    elif source == "Osiris":
        course = get_osiris_course(name, conn)

        if course is None:
            st.markdown("_Geen cursusdetails gevonden._")
            return

        # course = course.iloc[0]
        try:
            st.caption(f"## Vak: {course['lange_naam']} ({course['cursus']})")
        except Exception as e:
            print("NOOOO")
            pass
        # capt = f"## Vak: lnaam ({course['cursus']})"
        # st.caption(f"## Vakcode: {course['cursus']}")
        #
        # if pd.notna(course["lange_naam"]):
        #     # st.markdown(f"**Vaknaam:** {course['lange_naam']}")
        #     capt.replace("lnaam", str(course['lange_naam']))
        #     st.caption(capt)
        # else:
        #     st.caption(f"## Vakcode: {course['cursus']}")

        # st.caption(capt)
        # if pd.notna(course['docent_rol']):
        #     st.markdown(f"**DOCENTEN:** {course['docent_rol']}")

        # gekoppelde docenten

        # print("counts")
        # print(pd.read_sql("""
        #     SELECT COUNT(*)
        #     FROM course_employee
        #     """, conn))
        print("-------------------------")
        # print("course:", course)
        employees = get_employees_for_course(course["cursus"], conn)
        #
        # print(pd.read_sql("""
        # SELECT *
        # FROM course_employee
        # WHERE course = ?
        # LIMIT 5
        # """, conn, params=(course["cursus"],)))
        #
        # print("---")
        #
        # print("check")
        # print(pd.read_sql("""
        # SELECT *
        # FROM course_employee
        # LIMIT 10
        # """, conn))
        # print(employees)

        if not employees.empty:

            st.markdown("### Docenten")

            for _, employee in employees.iterrows():

                with st.expander(employee["employee"]):

                    if pd.notna(employee["employee_url"]):
                        st.markdown(f"[Medewerkerspagina]({employee['employee_url']})")

                    render_single_result(
                        {
                            "name": employee["employee"],
                            "source": "Employees"
                        },
                        conn)

        # overige cursusinformatie
        st.markdown("### Vak informatie")

        with st.expander("Meer info..."):

            if pd.notna(course["inhoud"]):
                st.markdown(f"**Inhoud:** {course['inhoud']}")

            if pd.notna(course["doel"]):
                st.markdown(f"**Doel:** {course['doel']}")


    elif source == "Repo":

        paper = get_repository_record(name, conn)

        if paper is None:
            st.markdown("_Geen publicatiegegevens gevonden._")
            return

        # paper = paper.iloc[0]

        st.caption(f"## {paper['title']}")

        if pd.notna(paper["authors"]):
            st.markdown(f"**Auteurs:** {paper['authors']}")

        if pd.notna(paper["department"]):
            st.markdown(f"**Afdeling:** {paper['department']}")

        if pd.notna(paper["keywords"]):
            st.markdown(f"**Keywords:** {paper['keywords']}")

        if pd.notna(paper["title_url"]):
            st.markdown(f"[Publicatie bekijken]({paper['title_url']})")


    else:
        st.warning(f"Onbekende bron: {source}")

    return


# ----------------------------
# UI building
st.set_page_config(page_title="Expert Finder", page_icon="🔎", layout="wide")


# if "selected_terms" not in st.session_state:
#     st.session_state.selected_terms = []

# st.sidebar.header("Plotting Demo")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


logo_base64 = get_base64_image("../app_layouts/RU_LOGO_COMPLEET.png")

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
st.info(
    "EN: This tool is the first version of a tool created for Radboud Universiteit - Onderwijs voor Professionals (OvP).\n"
    "It works best when using singular keywords such as 'ethics' or 'artificial intelligence'.\n"
    "Try to avoid full phrases like 'I\'m looking for ...'\n\n"
    "NL: Deze tool is de eerste versie van een ontwikkeling gedaan voor Radboud Universiteit - Onderwijs voor Professionals (OvP).\n"
    "Het werkt het beste op simpele trefwoorden zoals 'ethiek' of 'artificial intelligence'.\n"
    "Probeer zinnen zoals 'ik ben op zoek naar ...' te vermijden.", icon="⚠️"
)

# st.warning(
#         "NL: Deze tool is de eerste versie van een ontwikkeling gedaan voor Radboud Universiteit - Onderwijs voor Professionals (OvP).\n"
#         "Het werkt het beste op simpele trefwoorden zoals 'ethiek' of 'artificial intelligence'.\n"
#         "Probeer zinnen zoals 'ik ben op zoek naar ...' te vermijden.")

# pass


# AI-assistent voor Expert finder

st.subheader("AI-assistent")

if "ai_history" not in st.session_state:
    st.session_state.ai_history = []

if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

for message in st.session_state.ai_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Waar ben je naar op zoek?")

if query:
    history = st.session_state.ai_history.copy()

    st.session_state.ai_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Zoekresultaten ophalen en antwoord genereren..."):
            try:
                answer, results = answer_question(query, history, conn)
                st.markdown(answer)

                # Sla het antwoord ook op
                st.session_state.ai_history.append({"role": "assistant", "content": answer})
                st.session_state.ai_results = results

            except Exception as e:
                st.error(f"De AI-assistent kon geen antwoord geven: {e}")

if st.session_state.ai_results is not None:
    with st.expander("Bekijk de onderliggende zoekresultaten"):
        results = st.session_state.ai_results
        if isinstance(results, pd.DataFrame):
            st.dataframe(results, width="stretch")
        else:
            st.write(results)


# query = st.text_input("Zoekterm(en)/Search term(s)")
# TOP_FINAL = st.number_input("Max resultaten/results", min_value=1, max_value=150, value=10)

# with st.chat_message("assistant"):
#     st.write("Hello human")

# -----------------------------------
# FIXME
@st.dialog("Mijn Pop-up Venster")
def toon_popup():
    st.write("Dit is de inhoud van het pop-upvenster.")
    if st.button("Sluiten"):
        st.rerun()

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
#
# if query:
#     # expanded search
#     # dfs = run_search(query, TOP_FTS)
#     query = query.strip(".,;:'\"[{]}()}]-_*&^%$#@!`~\\|")
#     results = run_search(query, TOP_FTS, conn)
#
#     linkedin_url = create_linkedin_url(query)
#     st.link_button("Zoek professionals op LinkedIn →", linkedin_url)
#
#     if results.empty:
#         st.warning(
#             "Geen resultaten gevonden"
#         )
#         st.stop()
#
#     # if not dfs:
#     if results is None:
#         st.markdown("_Geen resultaten gevonden._")
#         st.warning("Keyword not found. Try a different one.", icon="❗❗❗")
#     else:
#         TAB_LIMITS = {
#             "All": TOP_FINAL * 2,
#             "Osiris": max(5, TOP_FINAL),
#             "Employees": max(5, TOP_FINAL),
#             "Repo": max(5, TOP_FINAL),
#         }
#
#         results = results.sort_values("final_score", ascending=False)
#
#         # per bron
#         results_O = results[results["source"] == "Osiris"].head(TAB_LIMITS["Osiris"])
#         results_E = results[results["source"] == "Employees"].head(TAB_LIMITS["Employees"])
#         results_R = results[results["source"] == "Repo"].head(TAB_LIMITS["Repo"])
#
#         # mix results
#         results_all = interleave(
#             [results_E, results_O, results_R],
#             TOP_FINAL
#         )
#
#         tabs = st.tabs(["All", "Osiris", "Employees", "Repository"])
#
#         with tabs[0]:
#             for _, row in results_all.iterrows():
#                 render_single_result(row, conn)
#                 st.markdown("---")
#
#         with tabs[1]:
#             for _, row in results_O.iterrows():
#                 render_single_result(row, conn)
#                 st.markdown("---")
#
#         with tabs[2]:
#             for _, row in results_E.iterrows():
#                 render_single_result(row, conn)
#                 st.markdown("---")
#
#         with tabs[3]:
#             for _, row in results_R.iterrows():
#                 render_single_result(row, conn)
#                 st.markdown("---")
#
#     # except Exception as e:
#     #     st.write("No matches found.")
#     #     st.warning("Invalid input", icon="❗")
#     #     # st.write(e)
print("-------------------------")
print("-------------------------")
print("-------------------------")

# streamlit run Hello.py --server.runOnSave true

"""
cd OvP
.venv\Scripts\activate
cd pages
uv run streamlit run 1_Expert_Finder.py --server.runOnSave true

"""
