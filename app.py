# import streamlit as st
# from openai import OpenAI
# from transformers import pipeline
# import torch as pt
#
# # Show title and description.
# st.title("💬 OvP - AI tool")
# st.write("This tool is used to provide an overview of teachers and professionals related to the enquired topic."
#          "")
#
# # to be donefinished at a later time, need specific and concrete examples examples
# st.info("Example:\n "
#         "If you are looking for an Artificial Intelligence (AI) specialist,"
#         " with a specific field domain such as 'data science', 'computer vision' etc. include that."
#         "\n\nThe more specific you are, the better the results will be ;)"
#         ,icon="🗝️")
#
# # st.write(
# #     "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
# #     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
# #     "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
# # )
#
#
# # Load the model (Example: GPT-2)
# model = pipeline('text-generation', model='gpt2')
#
# def generate_response(input_text):
#     return model(input_text, max_length=50, num_return_sequences=1)[0]['generated_text']
#
# prompt = st.chat_input("What kind of professional are you looking for?")
#
#
#
# # # Ask user for their OpenAI API key via `st.text_input`.
# # # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# # # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# # openai_api_key = st.text_input("OpenAI API Key", type="password")
# # if not openai_api_key:
# #     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# # else:
#
# #     # Create an OpenAI client.
# #     client = OpenAI(api_key=openai_api_key)
#
# #     # Create a session state variable to store the chat messages. This ensures that the
# #     # messages persist across reruns.
# #     if "messages" not in st.session_state:
# #         st.session_state.messages = []
#
# #     # Display the existing chat messages via `st.chat_message`.
# #     for message in st.session_state.messages:
# #         with st.chat_message(message["role"]):
# #             st.markdown(message["content"])
#
# #     # Create a chat input field to allow the user to enter a message. This will display
# #     # automatically at the bottom of the page.
# #     if prompt := st.chat_input("What kind of professional are you looking for?"):
#
# #         # Store and display the current prompt.
# #         st.session_state.messages.append({"role": "user", "content": prompt})
# #         with st.chat_message("user"):
# #             st.markdown(prompt)
#
# #         # Generate a response using the OpenAI API.
# #         stream = client.chat.completions.create(
# #             model="gpt-3.5-turbo",
# #             messages=[
# #                 {"role": m["role"], "content": m["content"]}
# #                 for m in st.session_state.messages
# #             ],
# #             stream=True,
# #         )
#
# #         # Stream the response to the chat using `st.write_stream`, then store it in
# #         # session state.
# #         with st.chat_message("assistant"):
# #             response = st.write_stream(stream)
# #         st.session_state.messages.append({"role": "assistant", "content": response})
import base64

import pandas as pd
import re
import spacy
import streamlit as st
from functools import lru_cache
from scraper import load_all_data

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
    keyword = st.text_input("Enter keyword")
    max_results = st.number_input("Max results", min_value=5, value=5)

    if st.button("Search"):
        scores = compute_scores(keyword, employee_data, repo_data, osiris_data)

        if not scores:
            st.write("No results found.")
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        st.subheader("Results")
        for name, score in ranked[:max_results]:
            st.write(f"**{name}** — {score} occurrences")

run_streamlit_ui(employee_data, repo_data, osiris_data)


