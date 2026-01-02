import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
    return _model

def embed_texts(texts):
    model = get_model()
    embd = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embd