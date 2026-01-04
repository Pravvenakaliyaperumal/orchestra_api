from typing import List
from models.chunk_strategy import ChunkStrategy

def _get_splitter(strategy: ChunkStrategy):
    # For our MVP we support the exact splitter you showed:
    # RecursiveCharacterTextSplitter from langchain.text_splitter
    if strategy.chunk_splitter_class != "RecursiveCharacterTextSplitter":
        raise ValueError(
            f"Unsupported splitter class: {strategy.chunk_splitter_class}. "
            "MVP supports RecursiveCharacterTextSplitter only."
        )

    # Import from langchain (modern versions)
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(**strategy.chunk_splitter_kwargs)

def chunk_text(text: str, strategy: ChunkStrategy) -> List[str]:
    splitter = _get_splitter(strategy)
    return splitter.split_text(text)
