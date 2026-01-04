import os
from sentence_transformers import SentenceTransformer

from config.mongo import get_db
from models.chunk_strategy import ChunkStrategy
from config.settings import TOP_K
from services.retriever import retrieve_top_k
from services.prompt_builder import build_rag_prompt

def embed_query_local(text: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text).tolist()

def main():
    db = get_db()
    strategy_id = os.getenv("CHUNK_STRATEGY_ID", "4")

    strategy_doc = db["chunk_strategy"].find_one({"_id": strategy_id})
    if not strategy_doc:
        raise RuntimeError(f"chunk_strategy not found for _id={strategy_id}")

    strategy = ChunkStrategy.from_mongo(strategy_doc)

    user_query = input("Enter your question: ").strip()
    if not user_query:
        print("No query provided.")
        return

    # ✅ LOCAL query embedding (same model as chunks)
    qvec = embed_query_local(user_query)

    top_chunks = retrieve_top_k(
        db,
        query_vector=qvec,
        strategy_id=strategy.chunk_strategy,
        k=TOP_K,
    )

    print(f"\n✅ Retrieved {len(top_chunks)} chunks (top_k={TOP_K})")
    for i, c in enumerate(top_chunks, start=1):
        print(f"\n--- CHUNK {i} ---")
        print(c["_id"])
        print(c["content"]["text"][:300])

    prompt = build_rag_prompt(user_query, top_chunks)

    print("\n====================")
    print("RAG PROMPT (sent to LLM)")
    print("====================")
    print(prompt)

    print("\n⚠️ LLM call skipped (no API dependency).")
    print("You can now plug this prompt into ANY LLM.")

if __name__ == "__main__":
    main()
