import base64
import sys


# import pandas as pd
import streamlit as st
# import numpy as np
from pathlib import Path
import streamlit.components.v1 as components

cwd = Path.cwd()

print("cwd:", cwd)

# from io import BytesIO
# import requests
from .OvP.database import *
# from OvP.utils import *
# from OvP.search import *


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# ----------------------------
# UI building
st.set_page_config(page_title="", layout="wide", page_icon="👋")
if "selected_terms" not in st.session_state:
    st.session_state.selected_terms = []

if "global_data" not in st.session_state:
    # Load your heavy data or configuration here
    # st.session_state.global_data = pd.read_csv("large_dataset.csv")
    embeddings, meta, model = load_resources()
    conn = get_connection()
    pass
# st.sidebar.success("Select a demo above.")

# pages_folder = Path("app")
#
# app = []
#
# for file in sorted(pages_folder.glob("*.py")):
#     # print(file.stem)
#     st.write(file.stem)
#     app.append(
#         st.Page(
#             str(file),
#             title=file.stem.replace("_", " ").title(),
#         )
#     )
#
# pg = st.navigation(app, position="sidebar")
#
# pg.run()


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

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# st.page_link("pages/1_Expert_Finder.py", label="Get Started!", width="stretch")
# if st.button("Get started!", width=300):
#
#     pass

# Inject custom CSS to increase padding, font size, and dimensions
st.markdown(
    """
    <style>
    .stLinkButton a {
        font-size: 28px !important;
        padding: 12px 24px !important;
        width: 100% !important;
        background: 305CDE;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
_, col2, _ = st.columns([1, 2, 1])

with col2:
    st.link_button("Get started!", "pages/1_Expert_Finder", type="primary")


with st.expander("AI modus"):
    st.markdown(
        f""" 
        In 'AI mode' helpt onze chat assistent om tot de kern van je vraag te komen en 
        daardoor de gerichtere resultaten te krijgen.
        Probeer het eerst zonder onze AI modus, mocht er niet gewenste resultaten uit komen vraag het dan aan onze assistent.
        """)
    st.write("Attach warning about AI etc")
    # AI_mode = st.toggle("AI mode")
    #
    # if AI_mode:
    #     st.write("AI mode activated!")
        # st.markdown(
        #     """
        #     <style>
        #     .reportview-container {
        #         background: url("url_goes_here")
        #     }
        #    .sidebar .sidebar-content {
        #         background: url("url_goes_here")
        #     }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )

        # js_code = """
        #         <script>
        #             window.open('pages/2_AI_Mode', '_blank').focus();
        #         </script>
        #     """
        # components.html(js_code, height=0)
        # st.write("Redirecting to link...")
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.link_button("Enter AI mode", "pages/2_AI_Mode", type="primary")
