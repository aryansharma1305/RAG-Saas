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
            file_path=str(target),
        )

    def ingest_url(self, url: str, workspace_id: str, kb_id: str) -> dict:
        self._validate_kb(workspace_id, kb_id)
        documents = self.parser.parse_url(url)
        return self._index_documents(
            documents=documents,
            workspace_id=workspace_id,
            kb_id=kb_id,
            source=url,
            file_path=None,
        )

    def _index_documents(
        self,
        documents,
        workspace_id: str,
        kb_id: str,
        source: str,
        file_path: str | None,
    ) -> dict:
        chunks = self.parser.to_chunks(documents)
        if not chunks:
            raise ValueError("No text chunks were extracted from the document")

        document_id = self.metadata.add_document(
            workspace_id=workspace_id,
            kb_id=kb_id,
            source=source,
            chunks_indexed=len(chunks),
            file_path=file_path,
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

    def delete_document(self, workspace_id: str, document_id: str) -> dict:
        document = self.metadata.get_document(workspace_id, document_id)
        if not document:
            raise ValueError("Document not found for this workspace")
        self.pinecone.delete_document(
            document_id=document["document_id"],
            chunks_indexed=int(document["chunks_indexed"]),
        )
        self.metadata.remove_document(workspace_id, document_id)
        return {
            "document_id": document["document_id"],
            "deleted_chunks": int(document["chunks_indexed"]),
        }

    def reindex_document(self, workspace_id: str, document_id: str) -> dict:
        document = self.metadata.get_document(workspace_id, document_id)
        if not document:
            raise ValueError("Document not found for this workspace")

        self._validate_kb(workspace_id, document["kb_id"])
        self.pinecone.delete_document(
            document_id=document["document_id"],
            chunks_indexed=int(document["chunks_indexed"]),
        )

        if document.get("file_path"):
            documents = self.parser.parse_file(Path(document["file_path"]))
        else:
            documents = self.parser.parse_url(document["source"])

        chunks = self.parser.to_chunks(documents)
        if not chunks:
            raise ValueError("No text chunks were extracted from the document")

        vectors = self._vectors_for_chunks(
            chunks=chunks,
            workspace_id=workspace_id,
            kb_id=document["kb_id"],
            document_id=document["document_id"],
            source=document["source"],
        )
        self.pinecone.upsert_chunks(vectors)
        self.metadata.update_document_chunks(document["document_id"], len(chunks))
        return {
            "document_id": document["document_id"],
            "chunks_indexed": len(chunks),
            "source": document["source"],
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
