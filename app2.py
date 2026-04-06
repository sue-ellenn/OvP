import base64
import streamlit as st
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
from pathlib import Path

# pd.read_csv("created_data/cleaned_data/repo.csv").to_parquet('created_data/cleaned_data/repo.parquet', compression="snappy")

FILES = {
    "search.db": "https://github.com/sue-ellenn/OvP/releases/download/data/search.db",
    "embeddings.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/embeddings.npy",
    "meta.npy": "https://github.com/sue-ellenn/OvP/releases/download/data/meta.npy"
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# @st.cache_resource
# def load_repo():
#     if not REPO_PATH.exists():
#         with st.spinner("Downloading repository data..."):
#             r = requests.get(REPO_URL)
#             r.raise_for_status()
#             REPO_PATH.write_bytes(r.content)
#     return pd.read_parquet(REPO_PATH)

# return pd.read_parquet(REPO_PATH)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = "search.db"
EMBEDDINGS_PATH = "created_data/DB_backups/embeddings.npy"

TOP_FTS = 100
TOP_FINAL = 0


def download_if_missing():
    for fname, url in FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            with st.spinner(f"Downloading {fname}..."):
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)


# download_if_missing()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_fts_query(user_query: str) -> str:
    terms = user_query.strip().split()
    if len(terms) == 1:
        return terms[0]
    exact = f'"{user_query}"'
    and_query = " AND ".join(terms)
    or_query = " OR ".join(terms)

    return f"{exact} OR ({and_query}) OR ({or_query})"


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


@st.cache_resource(show_spinner=False)
def load_resources():
    download_if_missing()
    conn = sqlite3.connect(DATA_DIR / "search.db", check_same_thread=False)
    emb = np.load(DATA_DIR / "embeddings.npy", mmap_mode="r")
    meta = np.load(DATA_DIR / "meta.npy", allow_pickle=True)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    R = pd.read_parquet("created_data/cleaned_data/repo.parquet")
    E = pd.read_csv("created_data/cleaned_data/employee.csv")
    O = pd.read_csv("created_data/cleaned_data/osiris.csv")

    return conn, emb, meta, model, E, O, R


conn, embeddings, meta, model, E, O, R = load_resources()


def run_search(query):
    fts_query = build_fts_query(query)

    sources = ["Employees", "Osiris", "Repo"]
    dfs = []

    q_emb = model.encode(query)

    for source in sources:
        df = pd.read_sql_query(
            """
            SELECT rowid, name, source, bm25(search) AS rank
            FROM search
            WHERE search MATCH ?
            AND source = ?
            ORDER BY rank
            LIMIT ?
            """,
            conn,
            params=(fts_query, source, TOP_FTS)
        )

        if df.empty:
            continue

        # BM25 normalization
        df["bm25_score"] = 1 / (1 + df["rank"])

        # Semantic similarity
        df["semantic_score"] = [
            cosine_sim(q_emb, embeddings[rowid - 1])
            for rowid in df["rowid"]
        ]

        # Exact match boost
        df["exact_match"] = df["name"].str.lower().str.contains(query.lower())

        # Final score
        df["final_score"] = (
                0.5 * df["semantic_score"] +
                0.4 * df["bm25_score"] +
                0.3 * df["exact_match"].astype(int)
        )

        # Length penalty (reduce repo dominance)
        df["final_score"] -= 0.0005 * df["name"].str.len()

        # Per source top 20
        df = df.sort_values("final_score", ascending=False).head(TOP_FINAL)

        dfs.append(df)

    if not dfs:
        st.write("BIG ERROR!!!!!!!!!!!!!!!!!!!!!")
        return pd.DataFrame()

    return dfs

    # except:
    #     return None


def expand_query_with_user_input(query, selected_terms):
    return " ".join([query] + selected_terms)


def get_query_suggestions(query, meta, embeddings, model, top_k=10):
    q_emb = model.encode(query)

    sims = np.dot(embeddings, q_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    )

    top_idx = np.argsort(sims)[-top_k:][::-1]

    suggestions = list({
        str(meta[i]["name"]) for i in top_idx
        if "name" in meta[i]
    })

    return suggestions


def get_suggestions_from_results(df):
    words = []

    for name in df["name"].head(20):
        words.extend(str(name).lower().split())

    words = [w for w in words if len(w) > 4]

    return list(set(words))[:10]


def interleave(dfs, max_total):
    result = []
    pointers = [0] * len(dfs)

    while len(result) < max_total:
        added = False

        for i, df in enumerate(dfs):
            if pointers[i] < len(df):
                result.append(df.iloc[pointers[i]])
                pointers[i] += 1
                added = True

                if len(result) >= max_total:
                    break

        if not added:
            break

    return pd.DataFrame(result)


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
    # Eerste snelle search voor suggestions
    initial_dfs = run_search(query)

    if initial_dfs:
        initial_results = pd.concat(initial_dfs)

        # Suggestions ophalen
        suggestions_sem = get_query_suggestions(query, meta, embeddings, model)
        suggestions_kw = get_suggestions_from_results(initial_results)

        suggestions = list(set(suggestions_sem + suggestions_kw))[:10]

        # selected_terms = st.multiselect(
        #             "Bedoelde je misschien / Related terms:",
        #             suggestions
        #         )
        # start
        st.markdown("**Bedoelde je misschien / Related terms:**")

        cols = st.columns(5)  # aantal blokjes per rij

        for i, term in enumerate(suggestions):
            col = cols[i % 5]

            is_selected = term in st.session_state.selected_terms

            if col.button(
                    term,
                    key=f"suggestion_{term}",
                    use_container_width=True
            ):
                if is_selected:
                    st.session_state.selected_terms.remove(term)
                else:
                    st.session_state.selected_terms.append(term)

        if st.session_state.selected_terms:
            st.write("Geselecteerd:", ", ".join(st.session_state.selected_terms))
        # stop

if query:
    # expanded_query = expand_query_with_user_input(query, selected_terms)
    expanded_query = expand_query_with_user_input(
        query,
        st.session_state.selected_terms
    )
    dfs = run_search(expanded_query)

    if not dfs:
        st.markdown("_Geen publicatiedetails gevonden._")
        st.warning("Keyword not found. Try a different one.", icon="❗❗❗")
    else:
        TAB_LIMITS = {
            "All": TOP_FINAL * 2,
            "Osiris": max(5, TOP_FINAL),
            "Employees": max(5, TOP_FINAL),
            "Repo": max(5, TOP_FINAL),
        }
        # st.write(results.columns)
        #
        # # Normalize BM25 rank (lower is better)
        # results["bm25_score"] = 1 / (1 + results["rank"])

        # print(results["bm25_score"])
        # print(results["semantic_score"])
        # st.columns(results["bm25_score"])

        # upscale employee db
        # for r in results:
        #     if r in ["rowid", "name"] :
        #         continue
        #     st.markdown(f"**{r}**")
        # if r["source"] == "Employees":
        #     r["bm25_score"] = r["bm25_score"] * 1.5


        # st.write(results["source"].value_counts())
        # If semantic_score exists, combine it
        # if "semantic_score" in results.columns:
        #     results["final_score"] = (
        #             0.6 * results["semantic_score"] +
        #             0.4 * results["bm25_score"]
        #     )
        # else:
        #     results["final_score"] = results["bm25_score"]

        # try:
        # combineer alle dataframes voor globale sortering indien nodig
        results = pd.concat(dfs).sort_values("semantic_score", ascending=False)

        # per bron
        results_E = next((df for df in dfs if df["source"].iloc[0] == "Employees"), pd.DataFrame())
        results_O = next((df for df in dfs if df["source"].iloc[0] == "Osiris"), pd.DataFrame())
        results_R = next((df for df in dfs if df["source"].iloc[0] == "Repo"), pd.DataFrame())

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
