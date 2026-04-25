# RAG SaaS Local Model MVP

FastAPI backend for a RAG SaaS pipeline using LlamaIndex parsing, Pinecone vector storage, local Ollama models, and metadata-scoped knowledge bases.

## What It Supports

- File ingestion: PDF, DOCX, TXT, CSV, and common LlamaIndex-supported files
- Web page ingestion by URL
- Pinecone serverless index creation
- Knowledge-base scoped retrieval
- Local model generation through Ollama
- Switchable model keys: `qwen`, `gemma`, `glm`

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Ollama from `https://ollama.com`, then pull small models that fit a 16GB RAM machine:

```bash
ollama pull qwen2.5:3b
ollama pull gemma3:1b
ollama pull glm4:9b
```

On Windows, after Ollama is installed and available in a new PowerShell window, you can pull the models configured in `.env` with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pull_ollama_models.ps1
```

If `glm4:9b` is too heavy, keep the config and use Qwen/Gemma first. You can change `QWEN_MODEL`, `GEMMA_MODEL`, and `GLM_MODEL` in `.env` when you have a different local model name available.

## Start API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## First Run Flow

1. Initialize Pinecone index:

```bash
python scripts/init_pinecone.py
```

2. Create a knowledge base:

```bash
curl -X POST http://127.0.0.1:8000/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"workspace_demo","name":"Demo KB"}'
```

3. Upload a document:

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload?workspace_id=workspace_demo&kb_id=YOUR_KB_ID" \
  -F "file=@/path/to/document.pdf"
```

4. Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"workspace_demo","kb_ids":["YOUR_KB_ID"],"question":"What is this document about?","model_key":"qwen"}'
```

## Knowledge Base Isolation

This app uses one Pinecone index and filters by metadata:

```json
{
  "workspace_id": "workspace_demo",
  "kb_id": "kb_uuid",
  "document_id": "doc_uuid",
  "source": "file.pdf"
}
```

Each chat/query chooses one or more `kb_ids`. Retrieval only runs against those KBs.

## Model Configuration

Edit `app/config.py` to change the local model names:

```python
qwen = "qwen2.5:3b"
gemma = "gemma3:1b"
glm = "glm4:9b"
```

The RAG code does not depend on one specific model, so benchmarking can reuse the same pipeline later.
