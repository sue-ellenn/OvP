import pandas as pd
from simple_app import compute_scores
from scraper import load_all_data

osiris_data, repo_data, employee_data = load_all_data()

def parse_keyword_file(path):
    """
    Basic parser for a TXT file where:
      - Keywords are before ':'
      - Employees are after ':'
      - Multiple keywords may be separated by ',' or '/'
      - Everything inside parentheses ( ... ) is removed
    Returns a list of dictionaries: { "keyword": k, "employee": e }
    """
    import re
    results = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            left, right = line.split(":", 1)

            # Remove text inside (...)
            left = re.sub(r"\(.*?\)", "", left).strip()

            # Split keywords and employees
            keywords = [k.strip() for k in re.split(r"[,/]", left) if k.strip()]
            employees = [e.strip() for e in re.split(r"[,/]", right) if e.strip()]

            for k in keywords:
                for e in employees:
                    results.append([k, e])

    return results

def test_keyword_employee_match(keyword, name):
    # keyword = "YOUR_KEYWORD"
    # expected_name = "EXPECTED_EMPLOYEE"


    scores = compute_scores(keyword, employee_data, repo_data, osiris_data)

    assert name in scores, f"Expected {name} not found for keyword {keyword}"
    assert scores[name] > 0




examples = parse_keyword_file("Voorbeeld zoektermen.txt")


for ex in examples:
    print("------------------------------------------------")
    print(f"Keyword: {ex[0]}, Employee: {ex[1]}")
    successes, failures = 0, 0
    try:
        test_keyword_employee_match(ex[0], ex[1])
        successes += 1
    except:
        print("Failed test")
        failures += 1
    print(f"Successes: {successes}/{failures}")
    print("------------------------------------------------")

    # lemma keywords
    # convert keywords into vectors
    # find closest matches
    # convert vectors back into keywords
    # get relevant names