import base64
import streamlit as st
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

DB_PATH = "search.db"
EMBEDDINGS_PATH = "created_data/DB_backups/embeddings.npy"

TOP_FTS = 100
TOP_FINAL = 20
MAX_PUBLICATIONS = 20
MAX_COURSES = 20


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_fts_query(user_query: str) -> str:
    terms = user_query.strip().split()
    if len(terms) == 1:
        return terms[0]
    phrase = f"\"{' '.join(terms)}\""
    and_query = " AND ".join(terms)
    return f"{phrase} OR ({and_query})"


def normalize_name(name):
    parts = name.lower().split()
    return {
        "initial": parts[0][0] if parts else "",
        "last": parts[-1] if parts else ""
    }


def docent_match(employee_name, docent_rol):
    if not isinstance(docent_rol, str):
        return False
    emp = normalize_name(employee_name)
    rol = docent_rol.lower()
    return emp["last"] in rol and emp["initial"] in rol


def get_themas(employee_row):
    themas = []
    for col in ["Keywords", "Onderzoeksthema", "Onderzoeksgroep"]:
        if col in employee_row and pd.notna(employee_row[col]):
            themas.extend(employee_row[col].split(","))
    return sorted(set(t.strip() for t in themas if len(t.strip()) > 2))


def get_publicaties(name, R, max_items=3):
    last_name = name.split()[-1].lower()
    pubs = R[R["authors"].str.lower().str.contains(last_name, na=False)]
    pubs = pubs.sort_values("publishing_info", ascending=False)
    return pubs[["title", "publishing_info", "title_url"]].head(max_items)


def get_docent_cursussen(name, O, max_items=3):
    mask = O["DOCENT_ROL"].apply(
        lambda x: docent_match(name, x)
    )
    return O[mask][["CURSUS", "LANGE_NAAM_NL", "DOEL"]].head(max_items)

def get_osiris_course(course_code, O):
    row = O[O["CURSUS"] == course_code]
    return None if row.empty else row.iloc[0]

def get_repository_record(title, R):
    row = R[R["title"] == title]
    return None if row.empty else row.iloc[0]


@st.cache_resource
def load_resources():
    conn = sqlite3.connect("search.db", check_same_thread=False)
    emb = np.load("created_data/DB_backups/embeddings.npy")
    meta = np.load("meta.npy", allow_pickle=True)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    R = pd.read_csv("created_data/cleaned_data/repo.csv")
    E = pd.read_csv("created_data/cleaned_data/employee.csv")
    O = pd.read_csv("created_data/cleaned_data/osiris.csv")

    return conn, emb, model, E, O, R


conn, embeddings, model, E, O, R = load_resources()


def run_search(query):
    fts_query = build_fts_query(query)

    sql = """
    SELECT rowid, name, source, bm25(search) AS rank
    FROM search
    WHERE search MATCH ?
    ORDER BY rank
    LIMIT ?
    """

    df = pd.read_sql_query(
        sql,
        conn,
        params=(fts_query, TOP_FTS)
    )

    if df.empty:
        return df

    q_emb = model.encode(query)

    df["semantic_score"] = [
        cosine_sim(q_emb, embeddings[rowid - 1])
        for rowid in df["rowid"]
    ]

    return df.sort_values(
        "semantic_score", ascending=False
    ).head(TOP_FINAL)

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

        with st.expander("Meer informatie"):
            st.markdown(f"**Inhoud:** {course['INHOUD']}")
            st.markdown(f"**Doel:** {course['DOEL']}")

    # repo
    elif source == "Research":
        rec = get_repository_record(name, R)
        st.write(rec.columns)

        if rec is None:
            st.markdown("_Geen publicatiedetails gevonden._")
            return

        # Title
        st.markdown(f"### {rec['title']}")

        # Authors
        if pd.notna(rec.get("authors")):
            st.markdown(f"**Auteurs:** {rec['authors']}")

        # Department
        if pd.notna(rec.get("department")):
            st.markdown(f"**Afdeling:** {rec['department']}")

        # Keywords
        if pd.notna(rec.get("keywords")):
            st.markdown(f"**Trefwoorden:** {rec['keywords']}")

        # Publication info
        if pd.notna(rec.get("publishing_info")):
            st.markdown(f"**Publicatie:** {rec['publishing_info']}")

        # Link
        if pd.notna(rec.get("title_url")):
            st.markdown(f"[Bekijk publicatie]({rec['title_url']})")

        # course = get_repo(name, O)




# ----------------------------
# UI building
st.set_page_config(page_title="", layout="wide")


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


query = st.text_input("Zoekterm(en)/\nSearch term(s)")

if query:
    # sql = """
    # SELECT rowid, name, source, bm25(search) AS rank
    # FROM search
    # WHERE search MATCH ?
    # ORDER BY rank
    # LIMIT 100
    # """
    #
    # df = pd.read_sql_query(sql, conn, params=(query,))
    # q_emb = model.encode(query)
    #
    # df["semantic_score"] = [
    #     np.dot(q_emb, embeddings[rowid - 1])
    #     for rowid in df["rowid"]
    # ]
    #
    # df = df.sort_values("semantic_score", ascending=False).head(20)
    #
    # for _, r in df.iterrows():
    #     st.markdown(f"""
    #     ### {r['name']}
    #     **Bron:** {r['source']}
    #     """)
    results = run_search(query)

    TAB_LIMITS = {
        "All": 20,
        "Osiris": 5,
        "Employees": 5,
        "Repo": 5,
    }
    # st.write(results.columns)

    # Normalize BM25 rank (lower is better)
    results["bm25_score"] = 1 / (1 + results["rank"])

    # If semantic_score exists, combine it
    if "semantic_score" in results.columns:
        results["final_score"] = (
                0.6 * results["semantic_score"] +
                0.4 * results["bm25_score"]
        )
    else:
        results["final_score"] = results["bm25_score"]

    # Ensure ordering
    results = results.sort_values("semantic_score", ascending=False)

    # Per-source subsets
    results_O = results[results["source"] == "Osiris"].head(TAB_LIMITS["Osiris"])
    results_E = results[results["source"] == "Employees"].head(TAB_LIMITS["Employees"])
    results_R = results[results["source"] == "Repo"].head(TAB_LIMITS["Repo"])

    # Alles tab: interleaved, capped at 20
    results_all = (
        pd.concat([results_E, results_O, results_R])
        .sort_values("final_score", ascending=False)
        .head(TAB_LIMITS["All"])
    )

    if results.empty:
        st.warning("No matches found.")
    else:
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
