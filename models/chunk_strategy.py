from dataclasses import dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class ChunkStrategy:
    _id: str
    chunk_strategy: int
    embedding_model: str
    chunk_splitter_class: str
    chunk_splitter_module: str
    chunk_splitter_kwargs: Dict[str, Any]

    @staticmethod
    def from_mongo(doc: Dict[str, Any]) -> "ChunkStrategy":
        required = [
            "_id",
            "chunk_strategy",
            "embedding_model",
            "chunk_splitter_class",
            "chunk_splitter_module",
            "chunk_splitter_kwargs",
        ]
        missing = [k for k in required if k not in doc]
        if missing:
            raise ValueError(f"Chunk strategy doc missing keys: {missing}")

        return ChunkStrategy(
            _id=str(doc["_id"]),
            chunk_strategy=int(doc["chunk_strategy"]),
            embedding_model=str(doc["embedding_model"]),
            chunk_splitter_class=str(doc["chunk_splitter_class"]),
            chunk_splitter_module=str(doc["chunk_splitter_module"]),
            chunk_splitter_kwargs=dict(doc["chunk_splitter_kwargs"]),
        )
