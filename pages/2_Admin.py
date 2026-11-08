import streamlit as st

import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from io import BytesIO
import requests
import sys
import os


sys.path.append(str(Path(__file__).resolve().parent.parent))


from admin.auth import check_login, logout
from admin.validation import read_osiris_file
from admin.update_database import update_osiris

st.set_page_config(
    page_title="Admin | Expert finder",
    page_icon="⚙️",
    layout="wide"
)


# Login controleren
if not check_login():
    st.stop()


# Dashboard
st.title("Admin Dashboard")
st.write("Beheer de cursusdata van de RU OvP Expert finder.")

if st.button("Uitloggen"):
    logout()


st.divider()

st.header("Osiris data beheren")

st.write(
    "Upload een vernieuwd Osiris-bestand om bestaande cursussen "
    "bij te werken en nieuwe cursussen toe te voegen."
)


uploaded_file = st.file_uploader(
    "Upload Osiris bestand",
    type=["csv", "xlsx", "xls"],
    key="osiris_upload"
)


if uploaded_file:
    if st.button("Bestand controleren", type="primary"):

        try:
            df = read_osiris_file(uploaded_file)

            st.session_state["osiris_upload_data"] = df
            st.session_state["osiris_upload_name"] = uploaded_file.name

            st.success("Bestand succesvol gecontroleerd.")

        except Exception as e:
            st.error(f"Fout bij verwerken: {e}")
            st.session_state.pop("osiris_upload_data", None)


# Preview
if "osiris_upload_data" in st.session_state:

    df = st.session_state["osiris_upload_data"]

    st.subheader("Preview nieuwe data")

    col1, col2, col3 = st.columns(3)

    col1.metric("Aantal cursussen", len(df))
    col2.metric("Aantal kolommen", len(df.columns))
    col3.metric("Nieuwe bestandsnaam", st.session_state["osiris_upload_name"])

    st.dataframe(df.head(20), use_container_width=True)

    st.info(
        "Bestaande cursussen worden bijgewerkt. "
        "Nieuwe cursussen worden toegevoegd. "
        "Cursussen die niet in het bestand voorkomen blijven behouden."
    )

    confirm = st.checkbox(
        "Ik heb de gegevens gecontroleerd en wil de database bijwerken."
    )

    if st.button(
        "Database bijwerken",
        type="primary",
        disabled=not confirm
    ):

        try:
            with st.spinner("Database wordt bijgewerkt..."):

                result = update_osiris(df)

            st.success("Osiris database succesvol bijgewerkt!")

            col1, col2 = st.columns(2)
            col1.metric("Bijgewerkte cursussen", result["updated"])
            col2.metric("Nieuwe cursussen", result["inserted"])

            st.write("Databaseback-up:", result["backup"])

            del st.session_state["osiris_upload_data"]
            del st.session_state["osiris_upload_name"]

        except Exception as e:
            st.error(f"Database-update mislukt: {e}")
            st.warning(
                "Controleer de foutmelding. De database-update is "
                "teruggedraaid als deze tijdens de transactie is mislukt."
            )



"""
cd OvP
.venv\Scripts\activate
cd pages
uv run streamlit run 2_Admin.py --server.runOnSave true

"""
