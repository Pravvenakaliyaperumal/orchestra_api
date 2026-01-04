from typing import List, Dict, Any, Tuple
import numpy as np

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)

def retrieve_top_k(db, *, query_vector: List[float], strategy_id: int, k: int = 3) -> List[Dict[str, Any]]:
    """
    1) Read all embeddings for strategy from db.chunk_embedding
    2) Score by cosine similarity
    3) Take top-k chunk_ids
    4) Fetch chunk docs from db.chunk
    """
    emb_col = db["chunk_embedding"]
    chunk_col = db["chunk"]

    scored: List[Tuple[float, str]] = []

    for emb in emb_col.find({"chunk_strategy": strategy_id}):
        vec = emb.get("vector")
        if not vec:
            continue
        score = cosine_similarity(query_vector, vec)
        scored.append((score, emb["_id"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_ids = [cid for _, cid in scored[:k]]

    if not top_ids:
        return []

    chunks = list(chunk_col.find({"_id": {"$in": top_ids}}))

    # Preserve rank order (Mongo $in order is not guaranteed)
    chunk_map = {c["_id"]: c for c in chunks}
    return [chunk_map[cid] for cid in top_ids if cid in chunk_map]
