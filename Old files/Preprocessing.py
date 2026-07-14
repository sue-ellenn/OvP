# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from nltk.corpus import stopwords
# import yake
# from langdetect import detect
# from rake_nltk import Rake
# # import rake
#
# # import data
# repo_data = pd.read_csv("created_data/repository/emp_20251115_161743.csv")
# repo_data = repo_data.drop_duplicates()
# print(repo_data.head(5))
#
# stop_words_dict = {'en': stopwords.words('english'), 'nl': stopwords.words('dutch')}
#
#
# # yake function
# def yake_extract_keywords(text, lang='eng'):
#     kw_extractor = yake.KeywordExtractor(
#                             lan=lang,           # or appropriate language code
#                             n=3,                # max n-gram size (try 2-3)
#                             dedupLim=0.8,       # make deduplication stricter
#                             dedupFunc='levs',   # try 'levs' or 'jaro' instead of default 'seqm'
#                             windowsSize=2,      # context window size
#                             top=5              # how many keywords you want
#                         )
#     keywords = kw_extractor.extract_keywords(text)
#     return [kw for kw, score in keywords]
#
# def rake_extract_keywords(text, lang='en'):
#     # rake = RAKE()
#     rake = Rake(stopwords=stop_words_dict[lang])
#     rake.extract_keywords_from_text(text)
#     return rake.get_ranked_phrases()
#
#
# def preprocess(text):
#     # strip non keywords
#     # stop_words_dict = {'eng': stopwords.words('english'), 'nl': stopwords.words('dutch')}
#     lang = detect(text)
#
#     print(f"Language: {lang}")
#     yake_kw = yake_extract_keywords(text, lang)
#     print("yake: ", yake_kw)
#     rake_kw = rake_extract_keywords(text, lang)
#     print("rake: ", rake_kw)
#     return
#
#
# count = 0
# for i, row in repo_data.iterrows():
#     title = row['title']
#     preprocessed_title = preprocess(title)
#     count += 1
#     if count > 10:
#         break

"""
research_keyword_pipeline.py

A ready-made pipeline to:
1. Load publications and employees CSVs
2. Preprocess (English + Dutch)
3. Extract candidate keywords (KeyBERT or YAKE fallback)
4. Cluster/normalize keywords with HDBSCAN
5. Vectorize keywords & entities with SentenceTransformers
6. Build a FAISS index for fast nearest-neighbor search
7. Match publications <-> employees via cosine similarity

Usage:
- Put your CSVs in the same folder and set PUB_CSV and EMP_CSV
- Install requirements in `requirements.txt` or pip install as shown below

Requirements (pip):
    pip install pandas numpy scikit-learn sentence-transformers keybert yake spacy hdbscan faiss-cpu umap-learn
    # optionally: pip install en-core-web-sm nl-core-news-sm (or use `python -m spacy download en_core_web_sm`)

Notes:
- The script defaults to KeyBERT for best semantic keyphrase extraction. If KeyBERT fails or you prefer a statistical extractor, set USE_YAKE = True.
- For Dutch support, the script tries to use spaCy nl model. If not present it falls back to a simple tokenizer.
- Adjust parameters like TOP_K_KEYWORDS, CLUSTER_MIN_CLUSTER_SIZE to taste.

"""

import os
import re
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Text processing
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Keyphrase extraction
try:
    from keybert import KeyBERT
    _HAS_KEYBERT = True
except Exception:
    _HAS_KEYBERT = False
    print("KeyBERT not available — will use YAKE fallback if enabled.")

try:
    import yake
    _HAS_YAKE = True
except Exception:
    _HAS_YAKE = False
    print("YAKE not available — install 'yake' for an alternative extractor.")

# embeddings & clustering
from sentence_transformers import SentenceTransformer
import hdbscan
import umap

# FAISS for vector search
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    print("faiss-cpu not installed. Install to use fast ANN. Will fallback to brute-force search.")


# ------------------------- Configuration -------------------------
PUB_CSV = "created_data/CORRECT_DATA/emp_20251115_161743.csv"        # replace with your file
EMP_CSV = "created_data/CORRECT_DATA/employees20251115_161743.csv"             # replace with your file
LANGUAGE = "auto"                   # 'auto' to detect by field or 'en'/'nl'
USE_YAKE = False                      # fallback to YAKE if True
TOP_K_KEYWORDS = 12
KEYWORD_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
ENTITY_VECTOR_STRATEGY = "mean"     # 'mean' or 'tfidf-weighted'
FAISS_DIM_REDUCTION = None            # e.g. 256 to reduce embed dim with UMAP before FAISS
CLUSTER_MIN_CLUSTER_SIZE = 5


# ------------------------- Utilities -------------------------

def safe_load_spacy(lang_code: str):
    """Try to load a spaCy model for the given language. Fall back to a simple tokenizer if not present."""
    try:
        if lang_code.startswith("en"):
            return spacy.load("en_core_web_sm")
        elif lang_code.startswith("nl"):
            return spacy.load("nl_core_news_sm")
        else:
            return spacy.load("en_core_web_sm")
    except Exception:
        print(f"spaCy model for {lang_code} not found — using simple fallback tokenizer.")
        return None


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2]
    return tokens


# ------------------------- Preprocessing & field merging -------------------------

def merge_publication_fields(row: pd.Series) -> str:
    parts = []
    for f in ["title", "keywords", "department", "publishing_info"]:
        if f in row and pd.notna(row[f]):
            parts.append(str(row[f]))
    return " \n ".join(parts)


def merge_employee_fields(row: pd.Series) -> str:
    parts = []
    for f in ["Keywords", "Onderzoeksthema", "Onderzoeksgroep", "Publicaties", "Curriculum Vitae"]:
        if f in row and pd.notna(row[f]):
            parts.append(str(row[f]))
    return " \n ".join(parts)


# ------------------------- Keyword extraction -------------------------

def extract_keywords_keybert(text: str, model: SentenceTransformer, top_k=TOP_K_KEYWORDS) -> List[Tuple[str, float]]:
    if not _HAS_KEYBERT:
        raise RuntimeError("KeyBERT not available")
    kw_model = KeyBERT(model)
    results = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,3), stop_words='english', top_n=top_k)
    # results: list of (keyword, score)
    return results


def extract_keywords_yake(text: str, top_k=TOP_K_KEYWORDS) -> List[Tuple[str, float]]:
    if not _HAS_YAKE:
        raise RuntimeError("YAKE not available")
    extractor = yake.KeywordExtractor(lan="en", n=3, top=top_k)
    kws = extractor.extract_keywords(text)
    # yake returns (kw, score) where low score is better — invert to be comparable
    return [(kw, 1.0 / (score + 1e-9)) for kw, score in kws]


# ------------------------- Keyword normalization & clustering -------------------------

def normalize_keyword(k: str) -> str:
    k = k.strip().lower()
    k = re.sub(r"[^\w\s-]", "", k)
    k = re.sub(r"\s+", " ", k)
    return k


def cluster_keywords(all_keywords: List[str], embed_model: SentenceTransformer, min_cluster_size=CLUSTER_MIN_CLUSTER_SIZE):
    if not all_keywords:
        return {}, []
    kw_embeddings = embed_model.encode(all_keywords, show_progress_bar=True)
    # reduce dim for clustering
    reducer = umap.UMAP(n_components=64, random_state=42)
    emb_reduced = reducer.fit_transform(kw_embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(emb_reduced)

    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    # build representative name per cluster: choose the most central keyword (closest to centroid)
    cluster_representatives = {}
    for label, idxs in clusters.items():
        if label == -1:
            # -1 is noise; skip or keep as singletons
            continue
        centroid = np.mean([kw_embeddings[i] for i in idxs], axis=0)
        distances = [np.linalg.norm(kw_embeddings[i] - centroid) for i in idxs]
        rep_idx = idxs[int(np.argmin(distances))]
        cluster_representatives[label] = all_keywords[rep_idx]

    return cluster_representatives, labels


# ------------------------- Embeddings & Indexing -------------------------

def build_keyword_embeddings(keywords: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(keywords, show_progress_bar=True)


def build_faiss_index(embeddings: np.ndarray) -> Tuple[object, np.ndarray]:
    # returns (index, normalized_embeddings)
    emb = embeddings.astype('float32')
    # normalize for cosine similarity as inner product
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb


# ------------------------- Entity vector computation -------------------------

def compute_entity_vector_from_keywords(keyword_list: List[str], kw_to_emb: Dict[str, np.ndarray], strategy='mean', tfidf_vectorizer: TfidfVectorizer = None) -> np.ndarray:
    vecs = [kw_to_emb[k] for k in keyword_list if k in kw_to_emb]
    if not vecs:
        # fallback: zero vector
        some_dim = next(iter(kw_to_emb.values())).shape[0]
        return np.zeros(some_dim, dtype=float)
    if strategy == 'mean':
        return np.mean(vecs, axis=0)
    elif strategy == 'tfidf-weighted' and tfidf_vectorizer is not None:
        # assume the caller computed tfidf weights for these keywords
        weights = []
        for k in keyword_list:
            try:
                weights.append(tfidf_vectorizer.vocabulary_.get(k, 0.0))
            except Exception:
                weights.append(1.0)
        weights = np.array(weights).astype(float)
        weights = weights / (weights.sum() + 1e-9)
        weighted = np.average(vecs, axis=0, weights=weights)
        return weighted
    else:
        return np.mean(vecs, axis=0)


# ------------------------- Matching -------------------------

def match_entities(pub_vecs: np.ndarray, emp_vecs: np.ndarray, top_k=5) -> List[List[Tuple[int, float]]]:
    """Compute top_k matching employees for each publication. Returns indices and scores."""
    sims = cosine_similarity(pub_vecs, emp_vecs)
    top_matches = []
    for row in sims:
        idxs = np.argsort(row)[::-1][:top_k]
        top_matches.append([(int(i), float(row[i])) for i in idxs])
    return top_matches


# ------------------------- Main pipeline -------------------------

def run_pipeline(pub_csv=PUB_CSV, emp_csv=EMP_CSV, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)

    pubs = pd.read_csv(pub_csv)
    emps = pd.read_csv(emp_csv)

    # Merge text fields
    pubs['merged_text'] = pubs.apply(merge_publication_fields, axis=1)
    emps['merged_text'] = emps.apply(merge_employee_fields, axis=1)

    # prepare embedding model
    embed_model = SentenceTransformer(KEYWORD_EMBED_MODEL)

    # extract keywords per entity
    all_candidate_keywords = []
    pub_keywords = []
    emp_keywords = []

    for t in pubs['merged_text'].astype(str).tolist():
        if USE_YAKE and _HAS_YAKE:
            kw = extract_keywords_yake(t, TOP_K_KEYWORDS)
        elif _HAS_KEYBERT:
            kw = extract_keywords_keybert(t, embed_model, TOP_K_KEYWORDS)
        else:
            # fallback: split and take top TF-IDF
            kw = []
        # normalize only the textual part of tuple
        knorm = [normalize_keyword(k[0]) for k in kw]
        pub_keywords.append(list(dict.fromkeys(knorm)))
        all_candidate_keywords.extend(knorm)

    for t in emps['merged_text'].astype(str).tolist():
        if USE_YAKE and _HAS_YAKE:
            kw = extract_keywords_yake(t, TOP_K_KEYWORDS)
        elif _HAS_KEYBERT:
            kw = extract_keywords_keybert(t, embed_model, TOP_K_KEYWORDS)
        else:
            kw = []
        knorm = [normalize_keyword(k[0]) for k in kw]
        emp_keywords.append(list(dict.fromkeys(knorm)))
        all_candidate_keywords.extend(knorm)

    all_candidate_keywords = list(dict.fromkeys([k for k in all_candidate_keywords if k]))
    print(f"Total unique candidate keywords: {len(all_candidate_keywords)}")

    # cluster similar keywords to compress
    cluster_reps, labels = cluster_keywords(all_candidate_keywords, embed_model, CLUSTER_MIN_CLUSTER_SIZE)
    print(f"Cluster reps (sample): {list(cluster_reps.items())[:10]}")

    # Optionally map each keyword to its cluster representative
    kw_to_canonical = {}
    if labels is not None and len(labels) == len(all_candidate_keywords):
        for i, k in enumerate(all_candidate_keywords):
            lbl = int(labels[i])
            if lbl == -1:
                kw_to_canonical[k] = k
            else:
                kw_to_canonical[k] = cluster_reps.get(lbl, k)
    else:
        for k in all_candidate_keywords:
            kw_to_canonical[k] = k

    # compress entity keywords to canonical set
    pub_keywords_canon = [[kw_to_canonical.get(k, k) for k in kws] for kws in pub_keywords]
    emp_keywords_canon = [[kw_to_canonical.get(k, k) for k in kws] for kws in emp_keywords]

    # build embeddings for canonical keywords
    canonical_keywords = list(sorted(set(kw_to_canonical.values())))
    kw_embeddings = build_keyword_embeddings(canonical_keywords, embed_model)
    kw_to_emb = {k: kw_embeddings[i] for i, k in enumerate(canonical_keywords)}

    # entity vectors
    pub_vecs = np.vstack([compute_entity_vector_from_keywords(klist, kw_to_emb, strategy=ENTITY_VECTOR_STRATEGY) for klist in pub_keywords_canon])
    emp_vecs = np.vstack([compute_entity_vector_from_keywords(klist, kw_to_emb, strategy=ENTITY_VECTOR_STRATEGY) for klist in emp_keywords_canon])

    # save canonical keywords and embeddings
    pd.DataFrame({'keyword': canonical_keywords}).to_csv(os.path.join(out_dir, 'canonical_keywords.csv'), index=False)

    # optionally build FAISS index on canonical keywords (for nearest keyword lookup)
    if _HAS_FAISS:
        kw_index, emb_norm = build_faiss_index(kw_embeddings)
        faiss.write_index(kw_index, os.path.join(out_dir, 'kw_faiss.index'))

    # match pubs to employees
    matches = match_entities(pub_vecs, emp_vecs, top_k=10)

    # format and save matches
    rows = []
    for pi, matchlist in enumerate(matches):
        for emp_idx, score in matchlist:
            rows.append({
                'pub_index': pi,
                'pub_title': pubs.iloc[pi].get('title', ''),
                'emp_index': emp_idx,
                'emp_name': emps.iloc[emp_idx].get('Name', ''),
                'score': score
            })
    matches_df = pd.DataFrame(rows).sort_values(['pub_index','score'], ascending=[True, False])
    matches_df.to_csv(os.path.join(out_dir, 'pub_emp_matches.csv'), index=False)

    print(f"Saved matches to {os.path.join(out_dir, 'pub_emp_matches.csv')}")
    return {
        'pub_keywords': pub_keywords_canon,
        'emp_keywords': emp_keywords_canon,
        'canonical_keywords': canonical_keywords,
        'pub_vecs': pub_vecs,
        'emp_vecs': emp_vecs,
        'matches_df': matches_df
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pub_csv', default=PUB_CSV)
    parser.add_argument('--emp_csv', default=EMP_CSV)
    parser.add_argument('--out', default='output')
    args = parser.parse_args()
    run_pipeline(args.pub_csv, args.emp_csv, args.out)
