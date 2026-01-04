import os
from config.mongo import get_db
from models.chunk_strategy import ChunkStrategy
from services.chunker import chunk_text
from services.document_ingestor import normalize_chunks_for_mongo, upsert_chunks

def ensure_strategy_exists(db, strategy_id: str):
    """
    If your DB already has chunk_strategy '4' (as in your screenshot), this does nothing.
    If not found, it inserts a default strategy using RecursiveCharacterTextSplitter.
    """
    col = db["chunk_strategy"]
    existing = col.find_one({"_id": strategy_id})
    if existing:
        return

    col.insert_one(
        {
            "_id": strategy_id,
            "name": strategy_id,
            "chunk_strategy": int(strategy_id),
            "embedding_model": "text-embedding-3-large",
            "chunk_splitter_class": "RecursiveCharacterTextSplitter",
            "chunk_splitter_module": "langchain.text_splitter",
            "chunk_splitter_kwargs": {
                "chunk_size": 350,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", " ", ""],
            },
        }
    )
    print(f"✅ Inserted default chunk_strategy {strategy_id} (because it was missing).")

def main():
    db = get_db()

    # Strategy ID = "4" (matches your screenshot)
    strategy_id = os.getenv("CHUNK_STRATEGY_ID", "4")

    ensure_strategy_exists(db, strategy_id)

    strategy_doc = db["chunk_strategy"].find_one({"_id": strategy_id})
    if not strategy_doc:
        raise RuntimeError(f"chunk_strategy not found for _id={strategy_id}")

    strategy = ChunkStrategy.from_mongo(strategy_doc)

    # Document ingest (for MVP: local file)
    with open("sample_data/sample_doc.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Create a stable document_id like your system
    document_id = "clientele:demo_doc_1"

    chunks = chunk_text(text, strategy)

    chunk_docs = normalize_chunks_for_mongo(
        document_id=document_id,
        chunks=chunks,
        chunk_strategy_id=strategy.chunk_strategy,
        base_metadata={"source": "sample_data/sample_doc.txt"},
    )

    upsert_chunks(db, chunk_docs)

    print("✅ Pipeline completed")
    print(f"Document ID: {document_id}")
    print(f"Chunk strategy: {strategy.chunk_strategy} (strategy _id={strategy._id})")
    print(f"Chunks written: {len(chunk_docs)}")
    print("Check MongoDB collection: clientele.chunk")

if __name__ == "__main__":
    main()
