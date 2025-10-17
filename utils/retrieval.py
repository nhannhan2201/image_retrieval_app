import torch
import faiss
import numpy as np

def load_model():
    import open_clip
    model_name = 'ViT-B-32'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model = model.to('cpu')
    return model, tokenizer, preprocess

def text_to_embedding(model, tokenizer, text: str):
    inputs = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def query_images(model, tokenizer, index, text, top_k=5):
    emb = text_to_embedding(model, tokenizer, text).astype(np.float32)
    D, I = index.search(emb, top_k)
    return I[0], D[0]
