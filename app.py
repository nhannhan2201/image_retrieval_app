import os
import streamlit as st
import pandas as pd
import faiss
from PIL import Image

from utils.retrieval import load_model, query_images
from utils.ocr_search import search_ocr

st.set_page_config(page_title="🎥 Video Retrieval Demo", layout="wide")

# ======================
# 🔧 LOAD CÁC TÀI NGUYÊN
# ======================
@st.cache_resource
def load_model_and_index():
    model, tokenizer, preprocess = load_model()
    index = faiss.read_index("data/merged_index.index")
    return model, tokenizer, index

@st.cache_data
def load_metadata():
    return pd.read_csv("data/metadata.csv")

@st.cache_data
def load_ocr():
    try:
        return pd.read_csv("data/ocr_data.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["id", "content"])

model, tokenizer, index = load_model_and_index()
meta_df = load_metadata()
ocr_df = load_ocr()

# ======================
# 🧭 GIAO DIỆN CHÍNH
# ======================
st.title("🎥 Video Retrieval Demo")
st.write("Tìm kiếm keyframe trong video bằng **CLIP (SigLIP2)** hoặc **OCR text search**")

mode = st.radio("Chọn phương thức:", ["Visual Search (CLIP)", "OCR Search"])
query = st.text_input("🔍 Nhập truy vấn:", "")
top_k = st.slider("Số lượng kết quả hiển thị:", 1, 10, 5)

if st.button("Bắt đầu tìm kiếm"):
    if not query.strip():
        st.warning("⚠️ Vui lòng nhập truy vấn trước khi tìm kiếm.")
        st.stop()

    if mode.startswith("Visual"):
        try:
            ids, dists = query_images(model, tokenizer, index, query, top_k)
        except Exception as e:
            st.error(f"❌ Lỗi khi truy vấn FAISS: {e}")
            st.stop()

        cols = st.columns(top_k)
        for i, (idx, dist) in enumerate(zip(ids, dists)):
            if idx < len(meta_df):
                row = meta_df.iloc[idx]
                fp = row.get("full_path", "")

                # đảm bảo path hợp lệ trên Streamlit Cloud
                if not os.path.exists(fp):
                    fp = os.path.join("data", "keyframes", os.path.basename(fp))

                with cols[i]:
                    if os.path.exists(fp):
                        img = Image.open(fp)
                        st.image(img, caption=f"{row.get('image_id', idx)} (score={1/(1+dist):.3f})")
                    else:
                        st.error(f"Ảnh không tồn tại: {fp}")
            else:
                with cols[i]:
                    st.warning("⚠️ Không tìm thấy metadata cho index này.")
    else:
        # OCR Search
        results = search_ocr(ocr_df, query, top_k)
        if not results:
            st.info("Không tìm thấy kết quả OCR nào phù hợp.")
        else:
            st.write("### 🧾 Kết quả OCR:")
            for r in results:
                st.markdown(f"**[{r['similarity']:.2f}]** → `{r['content'][:150]}...`")
