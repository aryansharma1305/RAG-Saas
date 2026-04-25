import time

from pinecone import Pinecone, ServerlessSpec

from app.config import Settings


class PineconeStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required")
        self.client = Pinecone(api_key=settings.pinecone_api_key)

    def ensure_index(self) -> None:
        existing = {index.name for index in self.client.list_indexes()}
        if self.settings.pinecone_index_name in existing:
            return

        self.client.create_index(
            name=self.settings.pinecone_index_name,
            dimension=self.settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self.settings.pinecone_cloud,
                region=self.settings.pinecone_region,
            ),
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.time() + 120
        while time.time() < deadline:
            description = self.client.describe_index(self.settings.pinecone_index_name)
            if getattr(description.status, "ready", False):
                return
            time.sleep(3)
        raise TimeoutError(f"Pinecone index '{self.settings.pinecone_index_name}' was not ready in time")

    @property
    def index(self):
        return self.client.Index(self.settings.pinecone_index_name)

    def upsert_chunks(self, vectors: list[dict]) -> None:
        if vectors:
            self.index.upsert(vectors=vectors)

    def delete_document(self, document_id: str, chunks_indexed: int) -> None:
        if chunks_indexed <= 0:
            return
        ids = [f"{document_id}_{idx}" for idx in range(chunks_indexed)]
        self.index.delete(ids=ids)

    def query(
        self,
        vector: list[float],
        workspace_id: str,
        kb_ids: list[str],
        top_k: int,
    ) -> list[dict]:
        result = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter={
                "workspace_id": {"$eq": workspace_id},
                "kb_id": {"$in": kb_ids},
            },
        )
        return [match.to_dict() for match in result.matches]
