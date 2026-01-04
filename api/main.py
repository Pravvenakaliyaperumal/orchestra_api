from fastapi import FastAPI
from api.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from api.deps import get_db_and_strategy

from services.chunker import chunk_text
from services.document_ingestor import (
    normalize_chunks_for_mongo,
    upsert_chunks,
)
from services.embedder import embed_chunks_for_strategy
from services.retriever import retrieve_top_k
from services.prompt_builder import build_rag_prompt
from sentence_transformers import SentenceTransformer
# api call
app = FastAPI(
    title="Mini Orchestra RAG API",
    description="Chunking → Embedding → Retrieval → RAG",
    version="1.0.0",
)

# Load local embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
def ingest_document(payload: IngestRequest):
    db, strategy = get_db_and_strategy()

    # 1️⃣ Chunk
    chunks = chunk_text(payload.text, strategy)

    # 2️⃣ Normalize
    chunk_docs = normalize_chunks_for_mongo(
        document_id=payload.document_id,
        chunks=chunks,
        chunk_strategy_id=strategy.chunk_strategy,
        base_metadata={"source": "api"},
    )

    # 3️⃣ Store
    upsert_chunks(db, chunk_docs)

    # 4️⃣ Embed
    embed_chunks_for_strategy(db, strategy)

    return IngestResponse(
        document_id=payload.document_id,
        chunks_written=len(chunk_docs),
    )

@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    db, strategy = get_db_and_strategy()

    # 1️⃣ Embed query (local)
    query_vector = embedding_model.encode(payload.question).tolist()

    # 2️⃣ Retrieve top-K
    chunks = retrieve_top_k(
        db,
        query_vector=query_vector,
        strategy_id=strategy.chunk_strategy,
        k=payload.top_k,
    )

    # 3️⃣ Build prompt
    rag_prompt = build_rag_prompt(payload.question, chunks)

    return QueryResponse(
        question=payload.question,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=c["_id"],
                content=c["content"]["text"],
            )
            for c in chunks
        ],
        rag_prompt=rag_prompt,
    )
