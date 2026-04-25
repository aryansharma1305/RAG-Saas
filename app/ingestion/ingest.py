from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import Settings
from app.embeddings import EmbeddingService
from app.ingestion.parser import DocumentParser
from app.pinecone_store import PineconeStore
from app.storage.metadata_store import MetadataStore


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        parser: DocumentParser,
        embeddings: EmbeddingService,
        pinecone: PineconeStore,
        metadata: MetadataStore,
    ):
        self.settings = settings
        self.parser = parser
        self.embeddings = embeddings
        self.pinecone = pinecone
        self.metadata = metadata
        self.upload_dir = Path("data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def ingest_upload(self, file: UploadFile, workspace_id: str, kb_id: str) -> dict:
        self._validate_kb(workspace_id, kb_id)
        safe_name = Path(file.filename or "upload").name
        target = self.upload_dir / f"{uuid4().hex}_{safe_name}"
        content = await file.read()
        target.write_bytes(content)

        documents = self.parser.parse_file(target)
        return self._index_documents(
            documents=documents,
            workspace_id=workspace_id,
            kb_id=kb_id,
            source=safe_name,
        )

    def ingest_url(self, url: str, workspace_id: str, kb_id: str) -> dict:
        self._validate_kb(workspace_id, kb_id)
        documents = self.parser.parse_url(url)
        return self._index_documents(
            documents=documents,
            workspace_id=workspace_id,
            kb_id=kb_id,
            source=url,
        )

    def _index_documents(
        self,
        documents,
        workspace_id: str,
        kb_id: str,
        source: str,
    ) -> dict:
        chunks = self.parser.to_chunks(documents)
        if not chunks:
            raise ValueError("No text chunks were extracted from the document")

        document_id = self.metadata.add_document(
            workspace_id=workspace_id,
            kb_id=kb_id,
            source=source,
            chunks_indexed=len(chunks),
        )
        vectors = self._vectors_for_chunks(
            chunks=chunks,
            workspace_id=workspace_id,
            kb_id=kb_id,
            document_id=document_id,
            source=source,
        )
        self.pinecone.upsert_chunks(vectors)
        return {
            "document_id": document_id,
            "chunks_indexed": len(chunks),
            "source": source,
        }

    def _vectors_for_chunks(
        self,
        chunks: list[str],
        workspace_id: str,
        kb_id: str,
        document_id: str,
        source: str,
    ) -> list[dict]:
        embeddings = self.embeddings.embed_texts(chunks)
        vectors = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append(
                {
                    "id": f"{document_id}_{idx}",
                    "values": embedding,
                    "metadata": {
                        "workspace_id": workspace_id,
                        "kb_id": kb_id,
                        "document_id": document_id,
                        "source": source,
                        "chunk_index": idx,
                        "text": chunk[:6000],
                    },
                }
            )
        return vectors

    def _validate_kb(self, workspace_id: str, kb_id: str) -> None:
        if not self.metadata.get_kb(workspace_id, kb_id):
            raise ValueError("Knowledge base not found for this workspace")
