# RAG SaaS

FastAPI backend for a multi-tenant RAG workflow with file ingestion, Pinecone retrieval, local Ollama generation, and knowledge-base scoped access.

## Features

- File ingestion: PDF, DOCX, TXT, CSV, and common LlamaIndex-supported files
- Web page ingestion by URL
- Pinecone serverless index creation
- Knowledge-base scoped retrieval
- Local model generation through Ollama
- Switchable model keys: `qwen`, `gemma`, `glm`
- Optional API-key auth with workspace ownership checks
- Document listing, deletion, and reindexing
- Optional cross-encoder reranking
- Request IDs and structured request logs

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Ollama from `https://ollama.com`, then pull the configured local models:

```bash
ollama pull qwen3:8b
ollama pull gemma3:4b
ollama pull glm4:9b
```

On Windows, the helper script reads model names from `.env`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pull_ollama_models.ps1
```

Copy `.env.example` to `.env` and set the Pinecone API key before initializing the index. LlamaParse is disabled by default and only needs a key when `USE_LLAMA_PARSE=true`.

## Start API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## First Run

1. Initialize Pinecone index:

```bash
python scripts/init_pinecone.py
```

2. Create a knowledge base:

```bash
curl -X POST http://127.0.0.1:8000/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"workspace_acme","name":"Company Policies"}'
```

3. Upload a document:

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload?workspace_id=workspace_acme&kb_id=kb_xxx" \
  -F "file=@/path/to/document.pdf"
```

4. Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"workspace_acme","kb_ids":["kb_xxx"],"question":"What is this document about?","model_key":"gemma"}'
```

## Knowledge Base Isolation

This app uses one Pinecone index and filters by metadata:

```json
{
  "workspace_id": "workspace_acme",
  "kb_id": "kb_xxx",
  "document_id": "doc_xxx",
  "source": "file.pdf"
}
```

Each chat/query chooses one or more `kb_ids`. Retrieval only runs against those KBs.

## Auth

Local development runs without auth by default. To enforce API-key auth and workspace ownership checks, set:

```env
AUTH_ENABLED=true
APP_API_KEY=change-me
```

Authenticated requests must include:

```text
X-API-Key: change-me
X-User-Id: user_123
```

The first user to create or access a workspace owns it. Requests from a different `X-User-Id` receive `403`.

## Document Lifecycle

List indexed documents:

```bash
curl "http://127.0.0.1:8000/documents?workspace_id=workspace_acme&kb_id=kb_xxx"
```

Delete a document and its vectors:

```bash
curl -X DELETE "http://127.0.0.1:8000/documents/doc_xxx?workspace_id=workspace_acme"
```

Reindex a document from its stored upload path or URL:

```bash
curl -X POST "http://127.0.0.1:8000/documents/doc_xxx/reindex?workspace_id=workspace_acme"
```

## Model Configuration

Edit `.env` to change the local model names:

```env
DEFAULT_MODEL_KEY=gemma
QWEN_MODEL=qwen3:8b
GEMMA_MODEL=gemma3:4b
GLM_MODEL=glm4:9b
```

The RAG code does not depend on one specific model, so benchmarking can reuse the same pipeline later.

Reranking can improve retrieval quality on larger knowledge bases:

```env
RERANKING_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Benchmark RAG Models

After ingesting at least one document, benchmark all configured model keys against an evaluation file:

```bash
python scripts/benchmark_rag.py --workspace-id workspace_acme --kb-id kb_xxx --output results/benchmark.json
```

If `--kb-id` is omitted, the script uses the latest knowledge base for the workspace. The default evaluation file is `data/sample_eval_questions.json`; replace it with questions and expected keywords from your own documents for a more meaningful score.
