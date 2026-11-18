
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

from database import get_connection
from search import run_search
from e_prompt import E_PROMPT
from utils import extract_search_terms, get_employee, get_osiris_course, get_repository_record


TOP_FTS = 100
MAX_TOOL_RESULTS = 20
MAX_FIELD_LENGTH = 1500
MAX_TOOL_ROUNDS = 8


def get_client():
    return OpenAI(
        api_key=st.secrets["LLM_API_KEY"],
        base_url=st.secrets["LLM_BASE_URL"]
    )


def to_record(value):
    if isinstance(value, pd.DataFrame):
        return None if value.empty else value.iloc[0].to_dict()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return None


def clean_record(record):
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

    cleaned = clean_record(record) if record else {}

    return {
        "name": str(name),
        "source": str(source),
        "score": round(float(result.get("final_score", 0)), 4),
        "url": cleaned.get("url", ""),
        "details": cleaned
    }

# ---------------------------------------------------------
# DATABASE TOOLS
# ---------------------------------------------------------



TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_employees",
            "description": (
                "Zoek medewerkers op naam, expertise, keywords, "
                "onderzoeksthema, onderzoeksgroep, onderwijs of publicaties. "
                "Gebruik meerdere zoekopdrachten als verschillende termen "
                "of bredere onderzoeksvelden relevant kunnen zijn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Zoekterm of zoekvraag."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Aantal resultaten, maximaal 20."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_courses",
            "description": (
                "Zoek universitaire cursussen en onderwijs op titel, "
                "inhoud, doel, docent of expertisegebied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Zoekterm of zoekvraag."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Aantal resultaten, maximaal 20."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_publications",
            "description": (
                "Zoek publicaties in de repository op titel, auteur, "
                "keywords, onderwerp of onderzoeksgebied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Zoekterm of zoekvraag."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Aantal resultaten, maximaal 20."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_employee_details",
            "description": (
                "Haal het volledige medewerkersprofiel op van een "
                "medewerker die al gevonden is, inclusief beschikbare "
                "keywords, onderzoeksthema's, onderzoeksgroep, onderwijs, "
                "publicaties en overige profielinformatie."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Naam van de medewerker."
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
    }
]
def search_source_original(query, source, limit, conn):
    limit = max(1, min(int(limit), MAX_TOOL_RESULTS))

    search_terms = extract_search_terms(query)

    results = run_search(
        search_terms,
        TOP_FTS,
        conn,
        max_results_per_source=limit,
        sources=[source]
    )

    if results is None or results.empty:
        return [], pd.DataFrame()

    results = results[results["source"] == source].copy()

    if results.empty:
        return [], pd.DataFrame()

    results = results.sort_values(
        "final_score",
        ascending=False
    ).head(limit)

    context = [
        enrich_result(row, conn)
        for _, row in results.iterrows()
    ]

    return context, results

def search_source(query, source, limit, conn):
    search_terms = extract_search_terms(query)

    if isinstance(search_terms, str):
        terms = search_terms.split()
    else:
        terms = list(search_terms)

    queries = [query]

    # Zoek ook afzonderlijk op de belangrijkste termen
    if len(terms) > 1:
        queries.extend(terms)

    all_results = []

    for term in dict.fromkeys(queries):
        try:
            _, results = search_source_original(term, source, limit, conn)
            all_results.append(results)
        except Exception as e:
            print(f"Zoekfout voor {term}: {e}")

    if not all_results:
        return [], pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.drop_duplicates(subset=["name", "source"])

    combined = combined.sort_values(
        "final_score",
        ascending=False
    ).head(limit)

    context = [
        enrich_result(row, conn)
        for _, row in combined.iterrows()
    ]

    return context, combined

def execute_tool(tool_name, arguments, conn):
    if tool_name == "search_employees":
        context, results = search_source(
            arguments["query"],
            "Employees",
            arguments.get("limit", 10),
            conn
        )
        return context, results

    if tool_name == "search_courses":
        context, results = search_source(
            arguments["query"],
            "Osiris",
            arguments.get("limit", 10),
            conn
        )
        return context, results

    if tool_name == "search_publications":
        context, results = search_source(
            arguments["query"],
            "Repo",
            arguments.get("limit", 10),
            conn
        )
        return context, results

    if tool_name == "get_employee_details":
        record = to_record(
            get_employee(arguments["name"], conn)
        )

        if not record:
            return {
                "error": (
                    f"Geen medewerkersprofiel gevonden voor "
                    f"{arguments['name']}."
                )
            }, pd.DataFrame()

        return clean_record(record), pd.DataFrame()

    return {"error": f"Onbekende tool: {tool_name}"}, pd.DataFrame()


# ---------------------------------------------------------
# LLM WITH DATABASE TOOL CALLING
# ---------------------------------------------------------

def parse_structured_answer(content):
    """Zet de JSON-uitvoer van de LLM om naar een Python-dictionary."""
    if not content:
        return {}

    content = content.strip()

    # Verwijder eventuele Markdown-codeblokken
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}



def answer_question(query, history=None, conn=None):
    import json
    import pandas as pd

    if conn is None:
        conn = get_connection()

    client = get_client()
    history = history or []

    messages = [
        {"role": "system", "content": E_PROMPT}
    ]

    # Gespreksgeschiedenis toevoegen
    for message in history[-10:]:
        if message.get("role") in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": str(message.get("content", ""))
            })

    messages.append({
        "role": "user",
        "content": query
    })

    all_results = []

    # --------------------------------------------------
    # 1. Zoekfase: tools uitvoeren
    # --------------------------------------------------

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=st.secrets["LLM_MODEL"],
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2
        )

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            break

        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                tool_result, result_df = execute_tool(
                    tool_name,
                    arguments,
                    conn
                )

                if isinstance(result_df, pd.DataFrame):
                    if not result_df.empty:
                        all_results.append(result_df.copy())

                tool_content = json.dumps(
                    tool_result,
                    ensure_ascii=False,
                    default=str
                )

            except Exception as e:
                print(f"Tool error ({tool_name}): {e}")
                tool_content = json.dumps({
                    "error": str(e)
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content
            })

    # --------------------------------------------------
    # 2. Alle gevonden resultaten combineren
    # --------------------------------------------------

    if all_results:
        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        combined = combined.drop_duplicates(
            subset=["source", "name"]
        )

        if "final_score" in combined.columns:
            combined = combined.sort_values(
                "final_score",
                ascending=False
            )

        combined = combined.head(MAX_TOOL_RESULTS * 2)

    else:
        combined = pd.DataFrame()

    # --------------------------------------------------
    # 3. Kandidaten verrijken voor de LLM
    # --------------------------------------------------

    candidates = []

    if not combined.empty:
        for index, (_, row) in enumerate(combined.iterrows()):
            try:
                candidate = enrich_result(row, conn)

                candidate["id"] = f"result_{index}"
                candidates.append(candidate)

            except Exception as e:
                print(f"Enrichment error: {e}")

    candidate_ids = {
        candidate["id"]: candidate
        for candidate in candidates
    }

    # --------------------------------------------------
    # 4. Synthese: inhoudelijk rangschikken
    # --------------------------------------------------

    if not candidates:
        return {
            "introduction": (
                "Ik heb helaas geen relevante resultaten "
                "kunnen vinden in de beschikbare databronnen."
            ),
            "results": [],
            "conclusion": ""
        }, pd.DataFrame()

    synthesis_prompt = f"""
Je bent de AI-assistent van de Radboud Expert Finder.

Gebruikersvraag:
{query}

Hieronder staan kandidaten uit de interne databronnen.
Beoordeel ze op inhoudelijke relevantie voor de gebruikersvraag.

BELANGRIJKE INSTRUCTIES:
- Beoordeel de daadwerkelijke inhoud van de profielen,
  cursussen en publicaties.
- Kijk naar synoniemen, gerelateerde begrippen en
  inhoudelijke verbanden.
- Een medewerker hoeft niet letterlijk de zoekterm
  in zijn functietitel of naam te hebben.
- Gebruik uitsluitend de kandidaten hieronder.
- Verzin geen medewerkers, cursussen of publicaties.
- Rangschik op inhoudelijke relevantie, niet uitsluitend
  op de oorspronkelijke zoekscore.
- Neem maximaal {MAX_TOOL_RESULTS} resultaten op.
- Geef per resultaat een korte uitleg waarom het relevant is.

Geef uitsluitend geldige JSON terug in deze structuur:

{{
    "introduction": "Korte introductie van de resultaten.",
    "results": [
        {{
            "id": "result_0",
            "explanation": "Waarom dit resultaat relevant is."
        }}
    ],
    "conclusion": "Korte afsluitende tekst."
}}

Beschikbare kandidaten:
{json.dumps(candidates, ensure_ascii=False, default=str)}
"""

    answer = None

    try:
        synthesis_response = client.chat.completions.create(
            model=st.secrets["LLM_MODEL"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Je bent een zorgvuldige expertzoekassistent. "
                        "Geef uitsluitend geldige JSON terug."
                    )
                },
                {
                    "role": "user",
                    "content": synthesis_prompt
                }
            ],
            tools=TOOLS,
            tool_choice="none",
            temperature=0.2
        )

        content = synthesis_response.choices[0].message.content

        print("SYNTHESIS RESPONSE:", repr(content))

        if content:
            content = content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                content = content.rsplit("```", 1)[0].strip()

            answer = json.loads(content)

    except Exception as e:
        print("Synthesis error:", repr(e))

    # --------------------------------------------------
    # 5. Fallback als de synthese mislukt
    # --------------------------------------------------

    if not isinstance(answer, dict):
        answer = {}

    if not isinstance(answer.get("results"), list):
        answer["results"] = []

    if not answer.get("introduction"):
        answer["introduction"] = (
            "Ik heb relevante resultaten gevonden "
            "op basis van je zoekopdracht."
        )

    if not answer.get("conclusion"):
        answer["conclusion"] = (
            "Bekijk de resultaten voor meer informatie "
            "over de gevonden expertise."
        )

    # --------------------------------------------------
    # 6. Geselecteerde kandidaten omzetten naar DataFrame
    # --------------------------------------------------

    selected_results = []
    used_ids = set()

    for item in answer["results"]:
        if not isinstance(item, dict):
            continue

        candidate_id = item.get("id")

        if candidate_id not in candidate_ids:
            continue

        if candidate_id in used_ids:
            continue

        candidate = candidate_ids[candidate_id]

        selected_results.append({
            "id": candidate_id,
            "name": candidate["name"],
            "source": candidate["source"],
            "explanation": item.get(
                "explanation",
                "Relevant op basis van de zoekopdracht."
            )
        })

        used_ids.add(candidate_id)

    # Als de LLM geen geldige selectie teruggeeft,
    # behouden we alle gevonden kandidaten op zoekscore.
    if not selected_results:
        for candidate in candidates[:MAX_TOOL_RESULTS]:
            selected_results.append({
                "id": candidate["id"],
                "name": candidate["name"],
                "source": candidate["source"],
                "explanation": (
                    "Gevonden op basis van de zoekopdracht."
                )
            })

    # --------------------------------------------------
    # 7. Resultaten in de juiste volgorde teruggeven
    # --------------------------------------------------

    ordered_rows = []

    for selected in selected_results:
        matching = combined[
            (combined["name"] == selected["name"]) &
            (combined["source"] == selected["source"])
        ]

        if matching.empty:
            continue

        row = matching.iloc[0].to_dict()
        row["explanation"] = selected["explanation"]
        row["_candidate_id"] = selected["id"]

        ordered_rows.append(row)

    ordered_results = pd.DataFrame(ordered_rows)

    return answer, ordered_results
