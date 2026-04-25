from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import get_settings


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rank(self, question: str, matches: list[dict], top_n: int) -> list[dict]:
        pairs = [(question, match.get("metadata", {}).get("text", "")) for match in matches]
        scores = self.model.predict(pairs)
        ranked = []
        for match, score in zip(matches, scores):
            item = dict(match)
            item["rerank_score"] = float(score)
            ranked.append(item)
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:top_n]


@lru_cache
def get_reranker() -> Reranker | None:
    settings = get_settings()
    if not settings.reranking_enabled:
        return None
    return Reranker(settings.reranker_model)
