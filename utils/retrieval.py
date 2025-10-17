import torch
import faiss
import numpy as np

def load_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-SO400M-16-SigLIP2-384', pretrained='webli'
    )
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-16-SigLIP2-384')
    model.eval().to('cpu')
    return model, tokenizer, preprocess

def text_to_embedding(model, tokenizer, text):
    inputs = tokenizer(text)
    with torch.no_grad():
        text_features = model.encode_text(inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def query_images(model, tokenizer, index, text, top_k=5):
    emb = text_to_embedding(model, tokenizer, text)
    D, I = index.search(emb, top_k)
    return I[0], D[0]
