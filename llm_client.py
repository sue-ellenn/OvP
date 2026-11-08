
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

from database import get_connection
from search import run_search
from e_prompt import E_PROMPT
from utils import extract_search_terms, get_employee, get_osiris_course, get_repository_record


TOP_FTS = 30
MAX_CONTEXT_RESULTS = 12
MAX_FIELD_LENGTH = 1800


def get_client():
    return OpenAI(
        api_key=st.secrets["LLM_API_KEY"],
        base_url=st.secrets["LLM_BASE_URL"]
    )


def to_record(value):
    """Zet een database-uitkomst om naar één dictionary."""
    if isinstance(value, pd.DataFrame):
        return None if value.empty else value.iloc[0].to_dict()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return None


def clean_record(record):
    """Verwijder lege waarden en beperk de lengte van velden."""
    cleaned = {}
    for key, value in record.items():
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        value = str(value).strip()
        if value:
            cleaned[key] = value[:MAX_FIELD_LENGTH]
    return cleaned


def enrich_result(result, conn):
    """Haal aanvullende gegevens op voor een zoekresultaat."""
    name = result["name"]
    source = result["source"]

    if source == "Employees":
        record = to_record(get_employee(name, conn))
    elif source == "Osiris":
        record = to_record(get_osiris_course(name, conn))
    elif source == "Repo":
        record = to_record(get_repository_record(name, conn))
    else:
        record = None

    return {
        "name": str(name),
        "source": str(source),
        "score": round(float(result.get("final_score", 0)), 4),
        "details": clean_record(record) if record else {}
    }


def resolve_search_query(query, history, client):
    """Maak van een vervolg vraag een zelfstandige zoekvraag."""
    if not history:
        return query

    messages = [
        {
            "role": "system",
            "content": (
                "Je zet vervolgvragen om naar zelfstandige zoekvragen "
                "voor een interne zoekmachine van een universiteit. "
                "Gebruik de chatgeschiedenis om verwijzingen zoals "
                "'zij', 'die medewerkers' en 'deze cursussen' te begrijpen. "
                "Behoud de oorspronkelijke zoekintentie. "
                "Voeg geen nieuwe onderwerpen toe. "
                "Geef uitsluitend de zelfstandige zoekvraag terug."
            )
        },
        *history[-8:],
        {
            "role": "user",
            "content": query
        }
    ]

    try:
        response = client.chat.completions.create(
            model=st.secrets["LLM_MODEL"],
            messages=messages,
            temperature=0
        )
        resolved_query = response.choices[0].message.content.strip()
        return resolved_query or query
    except Exception as e:
        print(f"Zoekvraag kon niet worden verduidelijkt: {e}")
        return query


def get_search_context(query, conn):
    search_terms = extract_search_terms(query)
    print("Zoektermen:", search_terms)

    results = run_search(search_terms, TOP_FTS, conn)

    if results is None or results.empty:
        return [], results

    results = results.sort_values("final_score", ascending=False)
    context = [
        enrich_result(row, conn)
        for _, row in results.head(MAX_CONTEXT_RESULTS).iterrows()
    ]

    return context, results


def answer_question(query, history=None, conn=None):
    """Beantwoord een vraag met behulp van interne zoekresultaten."""
    if conn is None:
        conn = get_connection()

    client = get_client()

    # Gebruik de geschiedenis om vervolgvragen te verduidelijken
    search_query = resolve_search_query(query, history or [], client)
    print("Oorspronkelijke vraag:", query)
    print("Gebruikte zoekvraag:", search_query)

    context, results = get_search_context(search_query, conn)

    if not context:
        return (
            "Ik heb geen passende resultaten gevonden in de beschikbare "
            "interne gegevens. Probeer andere zoektermen of formuleer je "
            "vraag iets specifieker.",
            results
        )

    messages = [
        {"role": "system", "content": E_PROMPT}
    ]

    messages.extend((history or [])[-8:])

    messages.append({
        "role": "user",
        "content": (
            f"Vraag van de gebruiker:\n{query}\n\n"
            f"Voor de interne zoekopdracht is deze zoekvraag gebruikt:\n"
            f"{search_query}\n\n"
            "Gebruik de eerdere conversatie om de vraag goed te begrijpen. "
            "Gebruik uitsluitend de volgende interne zoekresultaten "
            "als feitelijke basis:\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2)}"
        )
    })

    response = client.chat.completions.create(
        model=st.secrets["LLM_MODEL"],
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content
    if not answer:
        answer = "Ik kon geen antwoord formuleren op basis van deze resultaten."

    return answer, results
