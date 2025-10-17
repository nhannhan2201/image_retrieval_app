import pandas as pd
from rapidfuzz import fuzz
import re

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def search_ocr(df, query, top_k=5):
    keyword = normalize_text(query)
    results = []
    for idx, row in df.iterrows():
        text = normalize_text(row["content"])
        score = fuzz.ratio(keyword, text)
        results.append({
            "id": row.get("id", idx),
            "similarity": score,
            "content": text
        })
    results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results_sorted[:top_k]
