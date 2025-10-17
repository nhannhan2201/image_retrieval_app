import torch
import faiss
import numpy as np
import open_clip

def load_model():
    """
    Load SigLIP2 model (ViT-SO400M-16-SigLIP2-384) dùng để tạo embedding.
    """
    model_name = 'ViT-SO400M-16-SigLIP2-384'
    pretrained = 'webli'

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    device = torch.device("cpu")  # Streamlit Cloud không có GPU
    model.eval().to(device)

    return model, tokenizer, preprocess


def text_to_embedding(model, tokenizer, text: str):
    """
    Convert text thành embedding vector (chuẩn hóa norm).
    """
    inputs = tokenizer([text])
    with torch.no_grad():
        text_features = model.encode_text(inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # đảm bảo float32 để FAISS xử lý đúng
    return text_features.cpu().numpy().astype(np.float32)


def query_images(model, tokenizer, index, text, top_k=5):
    """
    Tìm ảnh gần nhất với text query trong FAISS index.
    """
    emb = text_to_embedding(model, tokenizer, text)
    index_dim = index.d
    if emb.shape[1] != index_dim:
        raise ValueError(f"❌ Dimension mismatch: embedding={emb.shape[1]}, index={index_dim}")

    D, I = index.search(emb, top_k)
    return I[0], D[0]
