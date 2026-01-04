import os
from config.mongo import get_db
from models.chunk_strategy import ChunkStrategy

def get_db_and_strategy():
    db = get_db()

    strategy_id = os.getenv("CHUNK_STRATEGY_ID", "4")
    strategy_doc = db["chunk_strategy"].find_one({"_id": strategy_id})

    if not strategy_doc:
        raise RuntimeError(f"chunk_strategy not found for _id={strategy_id}")

    strategy = ChunkStrategy.from_mongo(strategy_doc)
    return db, strategy
