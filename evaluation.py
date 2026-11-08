
from pathlib import Path
import sqlite3
import re
import unicodedata
import pandas as pd
from rapidfuzz import fuzz
from search import run_search


BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_PATH = BASE_DIR / "data" / "search_examples.txt"
DB_PATH = BASE_DIR / "data" / "search.db"

MATCH_THRESHOLD = 85


def load_evaluation_data(path=EXAMPLES_PATH):
    examples = []

    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or ":" not in line:
                continue

            topics, names = line.split(":", 1)
            topics = [t.strip() for t in topics.split(r"[,/]") if t.strip()]
            names = [n.strip() for n in names.split(",") if n.strip()]

            if topics and names:
                examples.append({"topics": topics, "names": names})
            else:
                print(f"Regel {line_number} overgeslagen.")

    return examples


def create_topic_mapping(examples):
    mapping = {}

    for example in examples:
        for topic in example["topics"]:
            mapping.setdefault(topic, set()).update(example["names"])

    return mapping


def normalize_name(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9_]+", name.lower())

    titles = {"dr", "prof", "professor", "mr", "drs", "ir"}
    tokens = [t for t in tokens if t not in titles]

    # Initialen verwijderen en eventuele anonimiseringstekens negeren
    tokens = [t.replace("_", "") for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]

    return " ".join(sorted(tokens))


def compare_names(expected, found):
    expected = normalize_name(expected)
    found = normalize_name(found)

    if not expected or not found:
        return 0

    # Vergelijk ook gedeeltelijke overeenkomsten bij geanonimiseerde namen
    return max(
        fuzz.token_sort_ratio(expected, found),
        fuzz.partial_token_sort_ratio(expected, found)
    )


def evaluate_search(conn, top_fts=30):
    examples = load_evaluation_data()
    topic_mapping = create_topic_mapping(examples)
    evaluation_results = []

    for topic, expected_names in topic_mapping.items():
        print(f"\nZoeken naar: {topic}")

        results = run_search(topic, top_fts, conn)
        print("\n--- DIAGNOSE ---")
        print("Topic:", topic)
        print("Aantal resultaten:", len(results))

        if results.empty or "source" not in results.columns:
            employee_results = pd.DataFrame()
        else:
            employee_results = results[
                results["source"] == "Employees"
            ].copy()

        if not employee_results.empty:
            employee_results = employee_results.sort_values(
                "final_score", ascending=False
            ).reset_index(drop=True)

        found_names = [
            (name, index + 1)
            for index, name in enumerate(employee_results["name"])
        ] if not employee_results.empty else []

        for expected_name in sorted(expected_names):
            matches = [
                (compare_names(expected_name, found_name), position, found_name)
                for found_name, position in found_names
            ]

            best_score, position, matched_name = max(
                matches, default=(0, None, None)
            )

            found = best_score >= MATCH_THRESHOLD

            evaluation_results.append({
                "topic": topic,
                "expected_name": expected_name,
                "found": found,
                "position": position if found else None,
                "matched_name": matched_name if found else None,
                "match_score": round(best_score, 1)
            })

    df = pd.DataFrame(evaluation_results)

    if not df.empty:
        print("\n========== EVALUATIERESULTATEN ==========")
        print(df.to_string(index=False))

        found = df["found"].sum()
        total = len(df)

        print(f"\nTeruggevonden: {found}/{total}")
        print(f"Recall: {found / total:.1%}")

        output_path = BASE_DIR / "data" / "evaluation_results.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nResultaten opgeslagen in: {output_path}")

    return df


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    try:
        evaluate_search(conn, top_fts=30)
    finally:
        conn.close()
