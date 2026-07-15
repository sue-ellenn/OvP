import base64
import streamlit as st
import numpy as np
from pathlib import Path
from io import BytesIO
import requests
from database import *
from utils import *
from search import *

# from build_course_mapping import *

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
TOP_FINAL = 0

conn, embeddings, meta, model = load_resources()


def render_single_result(result, conn):
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

            for _, course in courses.iterrows():
                st.write(f"- {course['course']}")


    elif source == "Osiris":
        course = get_osiris_course(name, conn)

        if course.empty:
            st.markdown("_Geen cursusdetails gevonden._")
            return

        course = course.iloc[0]

        st.caption(f"Vakcode: {course['cursus']}")

        if pd.notna(course["lange_naam"]):
            st.markdown(f"**Vaknaam:** {course['lange_naam']}")

        # gekoppelde docenten

        employees = get_employees_for_course(course["cursus"], conn)

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

        with st.expander("Meer informatie over vak"):

            if pd.notna(course["inhoud"]):
                st.markdown(f"**Inhoud:** {course['inhoud']}")

            if pd.notna(course["doel"]):
                st.markdown(f"**Doel:** {course['doel']}")


    elif source == "Repo":

        paper = get_repo_paper(name, conn)

        if paper.empty:
            st.markdown("_Geen publicatiegegevens gevonden._")
            return

        paper = paper.iloc[0]

        st.markdown(f"### {paper['title']}")

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


# ----------------------------
# UI building
st.set_page_config(page_title="", layout="wide")
if "selected_terms" not in st.session_state:
    st.session_state.selected_terms = []


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
st.warning(
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


query = st.text_input("Zoekterm(en)/Search term(s)")
TOP_FINAL = st.number_input("Max resultaten/results", min_value=1, max_value=150, value=10)

selected_terms = []

if query:
    # expanded search
    dfs = run_search(query, TOP_FTS)

    results = run_search(query, TOP_FTS)

    if results.empty:
        st.warning(
            "Geen resultaten gevonden"
        )
        st.stop()

    # if not dfs:
    if dfs is None:
        st.markdown("_Geen publicatiedetails gevonden._")
        st.warning("Keyword not found. Try a different one.", icon="❗❗❗")
    else:
        TAB_LIMITS = {
            "All": TOP_FINAL * 2,
            "Osiris": max(5, TOP_FINAL),
            "Employees": max(5, TOP_FINAL),
            "Repo": max(5, TOP_FINAL),
        }

        results = dfs.sort_values("final_score", ascending=False)

        # per bron
        results_O = results[results["source"] == "Osiris"].head(TAB_LIMITS["Osiris"])
        results_E = results[results["source"] == "Employees"].head(TAB_LIMITS["Employees"])
        results_R = results[results["source"] == "Repo"].head(TAB_LIMITS["Repo"])

        # mix results
        results_all = interleave(
            [results_E, results_O, results_R],
            TOP_FINAL
        )

        tabs = st.tabs(["All", "Osiris", "Employees", "Repository"])

        with tabs[0]:
            for _, row in results_all.iterrows():
                render_single_result(row)
                st.markdown("---")

        with tabs[1]:
            for _, row in results_O.iterrows():
                render_single_result(row)
                st.markdown("---")

        with tabs[2]:
            for _, row in results_E.iterrows():
                render_single_result(row)
                st.markdown("---")

        with tabs[3]:
            for _, row in results_R.iterrows():
                render_single_result(row)
                st.markdown("---")
    # except Exception as e:
    #     st.write("No matches found.")
    #     st.warning("Invalid input", icon="❗")
    #     # st.write(e)

# streamlit run app2.py --server.runOnSave true
# uv run streamlit run app2.py --server.runOnSave true
