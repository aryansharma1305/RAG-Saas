from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile

from app.auth import RequestContext, get_request_context
from app.config import Settings, get_settings
from app.dependencies import (
    get_ingestion_service,
    get_metadata_store,
    get_pinecone_store,
    get_rag_chain,
)
from app.ingestion.ingest import IngestionService
from app.logging_config import RequestLoggingMiddleware, configure_logging
from app.pinecone_store import PineconeStore
from app.rag_chain import RagChain
from app.schemas import (
    DeleteDocumentResponse,
    DocumentOut,
    HealthResponse,
    IngestResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseOut,
    QueryRequest,
    QueryResponse,
    UrlIngestRequest,
)
from app.storage.metadata_store import MetadataStore

configure_logging()
app = FastAPI(title="RAG SaaS", version="0.1.0")
app.add_middleware(RequestLoggingMiddleware)


def handle_access_error(exc: PermissionError) -> HTTPException:
    return HTTPException(status_code=403, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "pinecone_index": settings.pinecone_index_name,
        "embedding_model": settings.embedding_model,
        "models": settings.local_models,
    }


@app.post("/admin/pinecone/init")
def init_pinecone(
    pinecone: PineconeStore = Depends(get_pinecone_store),
    context: RequestContext = Depends(get_request_context),
) -> dict:
    try:
        pinecone.ensure_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/knowledge-bases", response_model=KnowledgeBaseOut)
def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    metadata: MetadataStore = Depends(get_metadata_store),
    context: RequestContext = Depends(get_request_context),
) -> dict:
    try:
        return metadata.create_kb(
            workspace_id=payload.workspace_id,
            name=payload.name,
            owner_id=context.user_id,
        )
    except PermissionError as exc:
        raise handle_access_error(exc) from exc


@app.get("/knowledge-bases", response_model=list[KnowledgeBaseOut])
def list_knowledge_bases(
    workspace_id: str = Query(..., min_length=1),
    metadata: MetadataStore = Depends(get_metadata_store),
    context: RequestContext = Depends(get_request_context),
) -> list[dict]:
    try:
        metadata.assert_workspace_access(workspace_id, context.user_id)
    except PermissionError as exc:
        raise handle_access_error(exc) from exc
    return metadata.list_kbs(workspace_id=workspace_id)


@app.post("/documents/upload", response_model=IngestResponse)
async def upload_document(
    workspace_id: str = Query(..., min_length=1),
    kb_id: str = Query(..., min_length=1),
    file: UploadFile = File(...),
    ingestion: IngestionService = Depends(get_ingestion_service),
    metadata: MetadataStore = Depends(get_metadata_store),
    context: RequestContext = Depends(get_request_context),
) -> dict:
    try:
        metadata.assert_workspace_access(workspace_id, context.user_id)
        return await ingestion.ingest_upload(
            file=file,
            workspace_id=workspace_id,
            kb_id=kb_id,
        )
    except PermissionError as exc:
        raise handle_access_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/documents/url", response_model=IngestResponse)
def ingest_url(
    payload: UrlIngestRequest,
    ingestion: IngestionService = Depends(get_ingestion_service),
    metadata: MetadataStore = Depends(get_metadata_store),
    context: RequestContext = Depends(get_request_context),
) -> dict:
    try:
        metadata.assert_workspace_access(payload.workspace_id, context.user_id)
        return ingestion.ingest_url(
            url=str(payload.url),
            workspace_id=payload.workspace_id,
            kb_id=payload.kb_id,
        )
    except PermissionError as exc:
        raise handle_access_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(
    workspace_id: str = Query(..., min_length=1),
    kb_id: str | None = Query(default=None),
    metadata: MetadataStore = Depends(get_metadata_store),
    context: RequestContext = Depends(get_request_context),
) -> list[dict]:
    try:
        metadata.assert_workspace_access(workspace_id, context.user_id)
    except PermissionError as exc:
        raise handle_access_error(exc) from exc
    return metadata.list_documents(workspace_id=workspace_id, kb_id=kb_id)


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    workspace_id: str = Query(..., min_length=1),
    ingestion: IngestionService = Depends(get_ingestion_service),
    metadata: MetadataStore = Depends(get_metadata_store),
    context: RequestContext = Depends(get_request_context),
) -> dict:
    try:
        metadata.assert_workspace_access(workspace_id, context.user_id)
        return ingestion.delete_document(workspace_id=workspace_id, document_id=document_id)
    except PermissionError as exc:
        raise handle_access_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/documents/{document_id}/reindex", response_model=IngestResponse)
def reindex_document(
    document_id: str,
    workspace_id: str = Query(..., min_length=1),
    ingestion: IngestionService = Depends(get_ingestion_service),
    metadata: MetadataStore = Depends(get_metadata_store),
    context: RequestContext = Depends(get_request_context),
) -> dict:
    try:
        metadata.assert_workspace_access(workspace_id, context.user_id)
        return ingestion.reindex_document(workspace_id=workspace_id, document_id=document_id)
    except PermissionError as exc:
        raise handle_access_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
    rag: RagChain = Depends(get_rag_chain),
    metadata: MetadataStore = Depends(get_metadata_store),
    context: RequestContext = Depends(get_request_context),
) -> dict:
    model_key = payload.model_key or settings.default_model_key
    try:
        metadata.assert_workspace_access(payload.workspace_id, context.user_id)
        return rag.answer(
            workspace_id=payload.workspace_id,
            kb_ids=payload.kb_ids,
            question=payload.question,
            model_key=model_key,
            top_k=payload.top_k,
            min_score=payload.min_score,
        )
    except PermissionError as exc:
        raise handle_access_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/models")
def list_models(
    settings: Settings = Depends(get_settings),
    context: RequestContext = Depends(get_request_context),
) -> dict:
    return {"default_model_key": settings.default_model_key, "models": settings.local_models}
