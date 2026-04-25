from app.embeddings import EmbeddingService
from app.pinecone_store import PineconeStore
from app.reranking import Reranker


class Retriever:
    def __init__(self, embeddings: EmbeddingService, pinecone: PineconeStore, reranker: Reranker | None = None):
        self.embeddings = embeddings
        self.pinecone = pinecone
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        workspace_id: str,
        kb_ids: list[str],
        top_k: int,
        min_score: float,
    ) -> list[dict]:
        query_vector = self.embeddings.embed_query(question)
        matches = self.pinecone.query(
            vector=query_vector,
            workspace_id=workspace_id,
            kb_ids=kb_ids,
            top_k=top_k * 4 if self.reranker else top_k,
        )
        filtered = [match for match in matches if float(match.get("score", 0.0)) >= min_score]
        if self.reranker and filtered:
            return self.reranker.rank(question, filtered, top_k)
        return filtered[:top_k]
