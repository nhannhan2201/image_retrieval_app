import streamlit as st
import pandas as pd
import faiss
from PIL import Image
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
from utils.retrieval import load_model, query_images
from utils.ocr_search import search_ocr

# ================================
# Cấu hình trang
# ================================
st.set_page_config(page_title="Video Retrieval Demo", layout="wide")

# ================================
# Load dữ liệu
# ================================
@st.cache_resource
def load_all_data():
    # Load FAISS index
    index = faiss.read_index("data/merged_index.index")
    meta_df = pd.read_csv("data/metadata.csv")

    # Load CLIP model
    model, tokenizer, _ = load_model()

    # Load OCR data
    ocr_df = pd.read_csv("data/ocr_data.csv")

    return model, tokenizer, index, meta_df, ocr_df


# ================================
# Giao diện
# ================================
st.title("🎥 Video Retrieval Demo")
st.write("Tìm kiếm keyframe trong video bằng **CLIP (Visual)** hoặc **OCR text similarity (Text)**")

# Chọn chế độ tìm kiếm
mode = st.radio("Chọn phương thức tìm kiếm:", ["Visual Search (CLIP)", "OCR Search (Text)"])

# Load tất cả tài nguyên
model, tokenizer, index, meta_df, ocr_df = load_all_data()

# Ô nhập truy vấn và số kết quả
query = st.text_input("Nhập truy vấn:", "người đàn ông đang phát biểu")
top_k = st.slider("Số lượng kết quả", 1, 10, 5)

# ================================
# Thực thi tìm kiếm
# ================================
if st.button("🔍 Tìm kiếm"):
    if mode.startswith("Visual"):
        # --- Search bằng CLIP ---
        ids, distances = query_images(model, tokenizer, index, query, top_k)
        cols = st.columns(top_k)

        for i, (idx, dist) in enumerate(zip(ids, distances)):
            row = meta_df.iloc[idx]

            # Xử lý đường dẫn ảnh an toàn cho cả local và cloud
            img_path = row["full_path"]
            if not os.path.exists(img_path):
                filename = os.path.basename(img_path)
                # Trích lấy thư mục con, ví dụ: keyframes/L22_V001/L22_V001
                parts = row["full_path"].replace("\\", "/").split("/")
                folder = "/".join(parts[-3:-1])
                img_path = os.path.join("data", "keyframes", folder, filename)

            try:
                img = Image.open(img_path)
                with cols[i]:
                    st.image(img, caption=f"{row['image_id']} (score={1/(1+dist):.3f})")
            except Exception as e:
                with cols[i]:
                    st.error(f"❌ Lỗi khi mở ảnh: {img_path}")
                    st.text(str(e))

    else:
        # --- Search bằng OCR ---
        results = search_ocr(ocr_df, query, top_k)
        st.write("### 🧾 Kết quả OCR:")
        for r in results:
            st.markdown(f"**[{r['similarity']:.2f}]** → `{r['content'][:120]}...`")
