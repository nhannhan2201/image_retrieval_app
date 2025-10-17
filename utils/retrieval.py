import torch
import faiss
import numpy as np

def load_model():
    """
    Load CLIP model SigLIP2 (tối ưu cho inference).
    Dùng CPU để tránh lỗi khi chạy trên Streamlit Cloud.
    """
    import open_clip
    model_name = 'ViT-SO400M-16-SigLIP2-384'
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='webli'
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # Đảm bảo model ở chế độ eval và nằm trên CPU
    model.eval()
    model = model.to("cpu")

    return model, tokenizer, preprocess


def text_to_embedding(model, tokenizer, text: str):
    """
    Chuyển truy vấn text thành vector embedding (chuẩn hóa).
    """
    if not text.strip():
        raise ValueError("❌ Truy vấn rỗng — vui lòng nhập câu tìm kiếm!")

    inputs = tokenizer([text])  # bọc trong list để tránh lỗi tensor shape
    with torch.no_grad():
        text_features = model.encode_text(inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def query_images(model, tokenizer, index, text, top_k=5):
    """
    Truy vấn FAISS index bằng vector của text query.
    Trả về danh sách (ids, distances).
    """
    emb = text_to_embedding(model, tokenizer, text)
    D, I = index.search(emb.astype(np.float32), top_k)  # FAISS yêu cầu float32
    return I[0], D[0]
