import numpy as np

def retrieve_top_k(question, chunks, chunk_embs, k):
    if not chunks or chunk_embs is None:
        return []
    
    k = min(k, len(chunks))  
    scores = chunk_embs @ question                       
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i]), chunks[i]) for i in top_idx]