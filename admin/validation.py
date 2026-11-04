
import pandas as pd


REQUIRED_COLUMNS = [
    "CURSUS",
    "LANGE_NAAM_NL",
    "DOCENT_ROL",
    "LABEL"
]

SOURCE_TEXT_COLUMN = "OSS_ADF_UTILITY.HTML_TO_TEXT(H.INHOUD)"


def read_osiris_file(uploaded_file):
    filename = uploaded_file.name.lower()

    # 1. Bestand inlezen
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python", dtype=str)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, dtype=str)
    else:
        raise ValueError("Gebruik een CSV- of Excel-bestand.")

    # 2. Kolomnamen normaliseren
    df.columns = [str(col).strip().upper() for col in df.columns]

    # 3. Controleren of de benodigde kolommen aanwezig zijn
    required = REQUIRED_COLUMNS + [SOURCE_TEXT_COLUMN.upper()]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Verplichte kolommen ontbreken: {missing}")

    # 4. Lege rijen verwijderen
    df = df.dropna(how="all").copy()

    df["CURSUS"] = df["CURSUS"].fillna("").astype(str).str.strip()
    df["LABEL"] = df["LABEL"].fillna("").astype(str).str.strip().str.lower()

    if (df["CURSUS"] == "").any():
        raise ValueError("Er zijn records zonder cursuscode.")

    # 5. LABEL gebruiken om inhoud en doel te scheiden
    text_column = SOURCE_TEXT_COLUMN.upper()

    df["INHOUD"] = df.apply(
        lambda row: row[text_column]
        if row["LABEL"] == "inhoud" else None,
        axis=1
    )

    df["DOEL"] = df.apply(
        lambda row: row[text_column]
        if row["LABEL"] == "doel" else None,
        axis=1
    )

    # 6. Twee rijen per cursus samenvoegen tot één rij
    result = df.groupby("CURSUS", as_index=False).agg({
        "LANGE_NAAM_NL": "first",
        "DOCENT_ROL": "first",
        "INHOUD": lambda x: next(
            (v for v in x if pd.notna(v) and str(v).strip()), None
        ),
        "DOEL": lambda x: next(
            (v for v in x if pd.notna(v) and str(v).strip()), None
        )
    })

    # 7. Dubbele cursusrecords controleren
    if result["CURSUS"].duplicated().any():
        raise ValueError("Dubbele cursuscodes gevonden na samenvoegen.")

    # 8. Kolommen in gewenste volgorde teruggeven
    return result[
        ["CURSUS", "LANGE_NAAM_NL", "DOCENT_ROL", "INHOUD", "DOEL"]
    ]
