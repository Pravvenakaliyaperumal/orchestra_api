from datetime import datetime, timezone
from typing import List
from sentence_transformers import SentenceTransformer

from models.chunk_strategy import ChunkStrategy

def utc_now():
    return datetime.now(timezone.utc)

def embed_chunks_for_strategy(db, strategy: ChunkStrategy, *, limit: int | None = None):
    """
    Local embedding using sentence-transformers.
    No API key, no quota, runs fully offline.
    """
    chunk_col = db["chunk"]
    emb_col = db["chunk_embedding"]

    # You can map strategy.embedding_model → local model if needed
    model = SentenceTransformer("all-MiniLM-L6-v2")

    cursor = chunk_col.find({"chunk_strategy": strategy.chunk_strategy})
    if limit:
        cursor = cursor.limit(limit)

    count = 0
    for chunk in cursor:
        chunk_id = chunk["_id"]
        text = chunk.get("content", {}).get("text", "").strip()
        if not text:
            continue

        vector: List[float] = model.encode(text).tolist()

        emb_col.update_one(
            {"_id": chunk_id},
            {
                "$set": {
                    "chunk_id": chunk_id,
                    "document_id": chunk["document_id"],
                    "chunk_strategy": strategy.chunk_strategy,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "vector": vector,
                    "time_updated": utc_now(),
                },
                "$setOnInsert": {"time_created": utc_now()},
            },
            upsert=True,
        )
        count += 1

    return count
