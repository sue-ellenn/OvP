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

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = "search.db"
EMBEDDINGS_PATH = FILES["embeddings.npy"]

response = requests.get(EMBEDDINGS_PATH)
response.raise_for_status()

embeddings = np.load(BytesIO(response.content))
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

TOP_FTS = 100
TOP_FINAL = 0


conn, embeddings, meta, model, E, O, R = load_resources()



def render_single_result(row, E, O, R):
    name = row["name"]
    source = row["source"]

    if source == "Osiris":
        course = get_osiris_course(name, O)
        name = course['LANGE_NAAM_NL']

    st.markdown("### " + name)
    name = row["name"]
    st.caption(f"Bron: {source}")

    # employees
    if source == "Employees":
        emp = E[E["Name"] == name]
        if emp.empty:
            st.markdown("_Geen profiel gevonden._")
            return

        emp_row = emp.iloc[0]

        # themas
        themas = get_themas(emp_row)
        if themas:
            st.markdown("**Thema’s:** " + ", ".join(themas))

        # onderwijs
        cursussen = get_docent_cursussen(name, O)
        if not cursussen.empty:
            st.markdown("**Onderwijs:**")
            for _, c in cursussen.iterrows():
                st.markdown(f"- {c['LANGE_NAAM_NL']}")

        # publicaties
        pubs = get_publicaties(name, R)
        if not pubs.empty:
            st.markdown("**Recente publicaties:**")
            for _, p in pubs.iterrows():
                st.markdown(f"- [{p['title']}]({p['title_url']})")

    # osiris
    elif source == "Osiris":
        course = get_osiris_course(name, O)
        if course is None:
            st.markdown("_Geen cursusdetails gevonden._")
            return

        # st.markdown(f"**{course['LANGE_NAAM_NL']}**")
        st.caption(f"Vakcode: {course['CURSUS']}")
        st.markdown(f"**Docent(en):** {course['DOCENT_ROL']}")
        employees = get_employees_for_course(
            course["CURSUS"],
            conn
        )

        if employees:

            st.markdown("### Gerelateerde experts")

            for employee in employees:
                with st.expander(employee):
                    render_single_result(
                        pd.Series({
                            "name": employee,
                            "source": "Employees"
                        }),
                        E,
                        O,
                        R
                    )

        with st.expander("Meer informatie"):
            st.markdown(f"**Inhoud:** {course['INHOUD']}")
            st.markdown(f"**Doel:** {course['DOEL']}")

    # repo
    elif source == "Repo":
        rec = get_repository_record(name, R)
        # st.write(rec)

        if rec is None:
            st.markdown("_Geen publicatiedetails gevonden._")
            return

        # Title
        # st.markdown(f"### {rec['title']}")
        # # st.write(rec.columns)

        # Authors
        if rec.get('authors') is not None:
            st.markdown(f"**Auteurs:** {rec['authors']}")

        # Department
        if rec.get('department') is not None:
            st.markdown(f"**Afdeling:** {rec['department']}")

        # Keywords
        if rec.get("keywords") is not None:
            st.markdown(f"**Trefwoorden:** {rec['keywords']}")

        # Publication info
        if rec.get("publishing_info") is not None:
            st.markdown(f"**Publicatie:** {rec['publishing_info']}")

        # Link
        if rec.get("title_url") is not None:
            st.markdown(f"[Bekijk publicatie]({rec['title_url']})")

        # course = get_repo(name, O)


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
TOP_FINAL = st.number_input("Max resultaten/results", min_value=1, max_value=150, value=50)

selected_terms = []

if query:
    # expanded search
    dfs = run_search(query, conn, embeddings, meta, model, TOP_FTS)


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
                render_single_result(row, E, O, R)
                st.markdown("---")

        with tabs[1]:
            for _, row in results_O.iterrows():
                render_single_result(row, E, O, R)
                st.markdown("---")

        with tabs[2]:
            for _, row in results_E.iterrows():
                render_single_result(row, E, O, R)
                st.markdown("---")

        with tabs[3]:
            for _, row in results_R.iterrows():
                render_single_result(row, E, O, R)
                st.markdown("---")
    # except Exception as e:
    #     st.write("No matches found.")
    #     st.warning("Invalid input", icon="❗")
    #     # st.write(e)

# streamlit run app2.py --server.runOnSave true
