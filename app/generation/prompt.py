SYSTEM_PROMPT = """You are a careful RAG assistant.
Answer only from the provided context.
If the context does not contain enough information, say: "I don't have enough information in the selected knowledge base."
Do not use outside knowledge.
Include concise citations using the source labels provided in the context."""


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        text = metadata.get("text", "")
        context_blocks.append(f"[{idx}] Source: {source}\n{text}")

    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
