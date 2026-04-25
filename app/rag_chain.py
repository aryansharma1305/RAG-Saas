from app.generation.llm_adapter import LocalLLMAdapter
from app.generation.prompt import build_rag_prompt
from app.retrieval import Retriever


class RagChain:
    def __init__(self, retriever: Retriever, llm: LocalLLMAdapter):
        self.retriever = retriever
        self.llm = llm

    def answer(
        self,
        workspace_id: str,
        kb_ids: list[str],
        question: str,
        model_key: str,
        top_k: int,
        min_score: float,
    ) -> dict:
        chunks = self.retriever.retrieve(
            question=question,
            workspace_id=workspace_id,
            kb_ids=kb_ids,
            top_k=top_k,
            min_score=min_score,
        )
        if not chunks:
            return {
                "answer": "I don't have enough information in the selected knowledge base.",
                "model_key": model_key,
                "sources": [],
            }

        prompt = build_rag_prompt(question, chunks)
        llm_response = self.llm.generate(model_key=model_key, user_prompt=prompt)

        return {
            "answer": llm_response["content"],
            "model_key": model_key,
            "sources": [
                {
                    "text": match["metadata"].get("text", ""),
                    "source": match["metadata"].get("source", "unknown"),
                    "document_id": match["metadata"].get("document_id", ""),
                    "kb_id": match["metadata"].get("kb_id", ""),
                    "score": float(match.get("score", 0.0)),
                }
                for match in chunks
            ],
        }
