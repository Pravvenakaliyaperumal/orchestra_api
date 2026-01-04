from pydantic import BaseModel
from typing import List

class IngestRequest(BaseModel):
    document_id: str
    text: str

class IngestResponse(BaseModel):
    document_id: str
    chunks_written: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str

class QueryResponse(BaseModel):
    question: str
    retrieved_chunks: List[RetrievedChunk]
    rag_prompt: str
