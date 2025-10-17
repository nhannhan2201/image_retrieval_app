import os
import streamlit as st
import pandas as pd
import faiss
from PIL import Image
from utils.retrieval import load_model, query_images
from utils.ocr_search import search_ocr

st.set_page_config(page_title="Video Retrieval Demo", layout="wide")

@st.cache_resource
def load_model_and_index():
    model, tokenizer, preprocess = load_model()
    idx = faiss.read_index("data/merged_index.index")
    return model, tokenizer, idx

@st.cache_data
def load_metadata():
    return pd.read_csv("data/metadata.csv")

@st.cache_data
def load_ocr():
    return pd.read_csv("data/ocr_data.csv")

model, tokenizer, index = load_model_and_index()
meta_df = load_metadata()
ocr_df = load_ocr()

st.title("🎥 Video Retrieval Demo")
mode = st.radio("Chọn phương thức:", ["Visual Search (CLIP)", "OCR Search"])
query = st.text_input("Nhập truy vấn:", "")
top_k = st.slider("Kết quả", 1, 10, 5)

if st.button("Tìm kiếm"):
    if mode.startswith("Visual"):
        ids, dists = query_images(model, tokenizer, index, query, top_k)
        cols = st.columns(top_k)
        for i, (idx, dist) in enumerate(zip(ids, dists)):
            if idx < len(meta_df):
                row = meta_df.iloc[idx]
                # Xử lý đường dẫn ảnh
                fp = row.get("full_path", "")
                if not os.path.exists(fp):
                    fp = os.path.join("data", "keyframes", os.path.basename(fp))
                if os.path.exists(fp):
                    img = Image.open(fp)
                    with cols[i]:
                        st.image(img, caption=f"{row.get('image_id', idx)} (score={1/(1+dist):.3f})")
                else:
                    with cols[i]:
                        st.write("Ảnh không tìm thấy")
    else:
        results = search_ocr(ocr_df, query, top_k)
        st.write("### Kết quả OCR")
        for r in results:
            st.write(f"{r['similarity']:.2f} — {r['content'][:100]}")
