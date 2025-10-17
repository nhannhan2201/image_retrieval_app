import torch
import faiss
import numpy as np
import open_clip

def load_model():
    model_name = "ViT-B-16-SigLIP"  # ✅ bản nhẹ
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="laion400m_e32"
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # ✅ Dùng half precision + CPU để giảm RAM
    model = model.half().to("cpu")
    model.eval()
    return model, tokenizer, preprocess

def text_to_embedding(model, tokenizer, text: str):
    inputs = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)

def query_images(model, tokenizer, index, text, top_k=5):
    emb = text_to_embedding(model, tokenizer, text)
    D, I = index.search(emb, top_k)
    return I[0], D[0]
