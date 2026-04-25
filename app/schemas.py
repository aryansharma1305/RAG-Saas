from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class KnowledgeBaseCreate(BaseModel):
    workspace_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class KnowledgeBaseOut(BaseModel):
    kb_id: str
    workspace_id: str
    name: str


class UrlIngestRequest(BaseModel):
    workspace_id: str
    kb_id: str
    url: HttpUrl


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    source: str


class DocumentOut(BaseModel):
    document_id: str
    workspace_id: str
    kb_id: str
    source: str
    chunks_indexed: int
    created_at: str


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted_chunks: int


class QueryRequest(BaseModel):
    workspace_id: str
    kb_ids: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    model_key: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SourceChunk(BaseModel):
    text: str
    source: str
    document_id: str
    kb_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    model_key: str
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    pinecone_index: str
    embedding_model: str
    models: dict[str, str]
