from typing import List, Dict, Any

def build_rag_prompt(user_query: str, chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for c in chunks:
        txt = c.get("content", {}).get("text", "")
        meta = c.get("metadata", {})
        chunk_id = c.get("_id", "")
        context_parts.append(f"[{chunk_id}] {txt}")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say: "I don't have enough information in the provided context."

CONTEXT:
{context}

QUESTION:
{user_query}

ANSWER:
""".strip()

    return prompt
