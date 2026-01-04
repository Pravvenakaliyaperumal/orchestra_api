from datetime import datetime, timezone
from typing import Any, Dict, List

def utc_now():
    return datetime.now(timezone.utc)

def normalize_chunks_for_mongo(
    *,
    document_id: str,
    chunks: List[str],
    chunk_strategy_id: int,
    base_metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Produces documents aligned with what you see in MongoDB Compass:
    - _id: "<document_id>:<chunk_strategy_id>:<chunk_number>"
    - document_id
    - chunk_number
    - chunk_strategy
    - content: { "text": ... }
    - metadata: { ... }
    - number_of_chunks
    - time_created, time_updated
    """
    base_metadata = base_metadata or {}
    now = utc_now()
    total = len(chunks)

    out: List[Dict[str, Any]] = []
    for idx, chunk_text in enumerate(chunks, start=1):
        out.append(
            {
                "_id": f"{document_id}:{chunk_strategy_id}:{idx}",
                "document_id": document_id,
                "chunk_number": idx,
                "chunk_strategy": chunk_strategy_id,
                "content": {"text": chunk_text},
                "metadata": dict(base_metadata),
                "number_of_chunks": total,
                "time_created": now,
                "time_updated": now,
            }
        )
    return out

def upsert_chunks(db, chunk_docs):
    col = db["chunk"]

    for doc in chunk_docs:
        doc_id = doc["_id"]

        # Remove time_created from $set
        update_doc = dict(doc)
        time_created = update_doc.pop("time_created")

        col.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    **update_doc,
                    "time_updated": utc_now(),
                },
                "$setOnInsert": {
                    "time_created": time_created
                },
            },
            upsert=True,
        )

