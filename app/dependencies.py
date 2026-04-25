from functools import lru_cache

from app.config import get_settings
from app.embeddings import get_embedding_service
from app.generation.llm_adapter import LocalLLMAdapter
from app.ingestion.ingest import IngestionService
from app.ingestion.parser import DocumentParser
from app.pinecone_store import PineconeStore
from app.rag_chain import RagChain
from app.retrieval import Retriever
from app.storage.metadata_store import MetadataStore


@lru_cache
def get_metadata_store() -> MetadataStore:
    return MetadataStore(get_settings())


@lru_cache
def get_pinecone_store() -> PineconeStore:
    return PineconeStore(get_settings())


@lru_cache
def get_document_parser() -> DocumentParser:
    return DocumentParser(get_settings())


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        settings=get_settings(),
        parser=get_document_parser(),
        embeddings=get_embedding_service(),
        pinecone=get_pinecone_store(),
        metadata=get_metadata_store(),
    )


@lru_cache
def get_llm_adapter() -> LocalLLMAdapter:
    return LocalLLMAdapter(get_settings())


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(
        embeddings=get_embedding_service(),
        pinecone=get_pinecone_store(),
    )


@lru_cache
def get_rag_chain() -> RagChain:
    return RagChain(
        retriever=get_retriever(),
        llm=get_llm_adapter(),
    )
