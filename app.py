import streamlit as st
import pandas as pd
import faiss
from PIL import Image

from utils.retrieval import load_model, query_images
from utils.ocr_search import search_ocr

st.set_page_config(page_title="Video Retrieval Demo", layout="wide")

@st.cache_resource
def load_all_data():
    # Load FAISS & CLIP
    index = faiss.read_index("data/merged_index.index")
    meta_df = pd.read_csv("data/metadata.csv")
    model, tokenizer, _ = load_model()
    
    # Load OCR data
    ocr_df = pd.read_csv("data/ocr_data.csv")
    
    return model, tokenizer, index, meta_df, ocr_df

st.title("🎥 Video Retrieval Demo")
st.write("Tìm kiếm keyframe trong video bằng **CLIP** (Visual) hoặc **OCR text similarity**")

mode = st.radio("Chọn phương thức tìm kiếm:", ["Visual Search (CLIP)", "OCR Search (Text)"])

model, tokenizer, index, meta_df, ocr_df = load_all_data()

query = st.text_input("Nhập truy vấn:", "người đàn ông đang phát biểu")
top_k = st.slider("Số lượng kết quả", 1, 10, 5)

if st.button("🔍 Tìm kiếm"):
    if mode.startswith("Visual"):
        ids, distances = query_images(model, tokenizer, index, query, top_k)
        cols = st.columns(top_k)
        for i, (idx, dist) in enumerate(zip(ids, distances)):
            row = meta_df.iloc[idx]
            img = Image.open(row["full_path"])
            with cols[i]:
                st.image(img, caption=f"{row['image_id']} (score={1/(1+dist):.3f})")
    else:
        results = search_ocr(ocr_df, query, top_k)
        st.write("### 🧾 Kết quả OCR:")
        for r in results:
            st.markdown(f"**[{r['similarity']:.2f}]** → `{r['content'][:120]}...`")
