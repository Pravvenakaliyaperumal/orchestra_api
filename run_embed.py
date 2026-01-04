import os
from config.mongo import get_db
from models.chunk_strategy import ChunkStrategy
from services.embedder import embed_chunks_for_strategy

def main():
    db = get_db()
    strategy_id = os.getenv("CHUNK_STRATEGY_ID", "4")

    strategy_doc = db["chunk_strategy"].find_one({"_id": strategy_id})
    if not strategy_doc:
        raise RuntimeError(f"chunk_strategy not found for _id={strategy_id}")

    strategy = ChunkStrategy.from_mongo(strategy_doc)

    count = embed_chunks_for_strategy(db, strategy)
    print(f"✅ Embedded & stored vectors for {count} chunks (strategy={strategy.chunk_strategy})")
    print("Check collection: clientele.chunk_embedding")

if __name__ == "__main__":
    main()
