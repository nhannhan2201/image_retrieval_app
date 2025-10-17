import os
import streamlit as st
import pandas as pd
import faiss
from PIL import Image

from utils.retrieval import load_model, query_images
from utils.ocr_search import search_ocr

# --------------------------------------------
# ⚙️ Cấu hình trang
# --------------------------------------------
st.set_page_config(page_title="🎥 Video Retrieval Demo", layout="wide")

# --------------------------------------------
# 🧠 Load mô hình & dữ liệu
# --------------------------------------------
@st.cache_resource
def load_model_faiss():
    """Load CLIP model và FAISS index (cache theo session)"""
    model, tokenizer, _ = load_model()
    index = faiss.read_index("data/merged_index.index")
    return model, tokenizer, index

@st.cache_data
def load_metadata():
    """Load metadata (các thông tin ảnh)"""
    return pd.read_csv("data/metadata.csv")

@st.cache_data
def load_ocr_data():
    """Load dữ liệu OCR"""
    return pd.read_csv("data/ocr_data.csv")

# Load tất cả dữ liệu cần thiết
model, tokenizer, index = load_model_faiss()
meta_df = load_metadata()
ocr_df = load_ocr_data()

# --------------------------------------------
# 🖼️ Giao diện chính
# --------------------------------------------
st.title("🎥 Video Retrieval Demo")
st.write("Tìm kiếm keyframe trong video bằng **CLIP (Visual)** hoặc **OCR text similarity (Text)**")

mode = st.radio("🔍 Chọn phương thức tìm kiếm:", ["Visual Search (CLIP)", "OCR Search (Text)"])
query = st.text_input("Nhập truy vấn:", "người đàn ông đang phát biểu")
top_k = st.slider("Số lượng kết quả hiển thị:", 1, 10, 5)

# --------------------------------------------
# 🚀 Thực thi tìm kiếm
# --------------------------------------------
if st.button("🔍 Tìm kiếm"):
    if mode.startswith("Visual"):
        st.write("Đang truy vấn bằng mô hình CLIP...")
        ids, distances = query_images(model, tokenizer, index, query, top_k)
        cols = st.columns(top_k)
        for i, (idx, dist) in enumerate(zip(ids, distances)):
            if idx >= len(meta_df):
                continue
            row = meta_df.iloc[idx]
            
            # 🧩 Xử lý đường dẫn ảnh an toàn
            img_path = row.get("full_path", "")
            if not os.path.exists(img_path):
                # Thử tìm trong thư mục keyframes
                img_path = os.path.join("data/keyframes", os.path.basename(img_path))
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                with cols[i]:
                    st.image(img, caption=f"{row.get('image_id', idx)} (score={1/(1+dist):.3f})")
            else:
                with cols[i]:
                    st.warning(f"⚠️ Không tìm thấy ảnh: {img_path}")

    else:
        st.write("Đang tìm kiếm bằng OCR...")
        results = search_ocr(ocr_df, query, top_k)
        st.write("### 🧾 Kết quả OCR:")
        for r in results:
            content = str(r.get("content", ""))[:120]
            st.markdown(f"**[{r.get('similarity', 0):.2f}]** → `{content}...`")
