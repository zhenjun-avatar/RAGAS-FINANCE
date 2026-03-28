# RAG Testing — Finance RAG & Agent Pipeline

Node-centric RAG stack for **financial disclosures** (e.g. SEC EDGAR HTML + structured company facts): **hybrid retrieval** (dense vectors + sparse / BM25), **SQL-backed observations**, optional **reranking**, **Langfuse** tracing, and a small **Next.js** UI.

> 中文简介：面向财务文档的混合检索（向量 + 稀疏）与结构化 SQL 证据结合的 RAG / Agent 实验与工程样例。

## Features

- **Ingestion**: SEC-style filings (local `.htm` / sync paths), company-facts JSON alignment (`finance_accns`, form, filing dates).
- **Retrieval**: Qdrant dense index; sparse via **PostgreSQL** or **OpenSearch** (finance-tuned query profile).
- **Orchestration**: LlamaIndex-style retrieval pipeline, finance query planning, optional LangGraph-related flows (see codebase).
- **API**: FastAPI (`/agent/...`), document catalog & **document groups** for scoped Q&A.
- **Frontend**: Next.js app proxying to the agent API; document / group selection for asks.

## Repository layout

```
rag-testing/
├── docker-compose.rag.yml   # Postgres + Qdrant + OpenSearch (optional sparse)
├── requirements.txt         # Python deps for src/agent (install from repo root)
├── src/
│   ├── agent/               # Python backend (FastAPI, ingestion, RAG)
│   │   ├── api/server.py
│   │   ├── core/config.py   # env-driven settings
│   │   ├── scripts/run_sec_finance_pipeline.py
│   │   ├── tools/           # retrieval, finance SQL, vector store, etc.
│   │   └── env.example      # copy to .env (do not commit secrets)
│   └── frontend/            # Next.js UI
```

## Prerequisites

- **Docker** (recommended) for Postgres, Qdrant, and optionally OpenSearch.
- **Python 3.11+** for `src/agent` (virtualenv).
- **Node.js 18+** for `src/frontend`.
- API keys as needed: **embedding** (e.g. DashScope / OpenAI) and **LLM** (e.g. DeepSeek / Qwen / OpenAI / Anthropic) — see `src/agent/env.example`.

## Quick start

### 1. Start infrastructure (from repo root)

```bash
docker compose -f docker-compose.rag.yml up -d
```

Default host ports (see compose file comments):

- Postgres: `5433` → map to `DATABASE_URL` / `DB_PORT`
- Qdrant: `6433` → `QDRANT_PORT`
- OpenSearch: `9200` → enable with `SPARSE_BACKEND=opensearch`

### 2. Configure the agent

```bash
cp src/agent/env.example src/agent/.env
# Edit .env: DATABASE_URL, Qdrant, embedding + LLM keys, SPARSE_BACKEND, OpenSearch + OPENSEARCH_SPARSE_INDEX_FINANCE if used
```

### 3. Install & run the API (`src/agent`)

```bash
# From repository root
python -m venv src/agent/venv
src\agent\venv\Scripts\pip install -r requirements.txt

cd src/agent
venv\Scripts\python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

On Linux/macOS: `src/agent/venv/bin/pip install -r requirements.txt` and `venv/bin/python -m uvicorn ...` from `src/agent`.

### 4. Run the frontend (`src/frontend`)

```bash
cd src/frontend
npm install
# Point Next.js at your API (default often http://127.0.0.1:8000)
npm run dev
```

Set `BACKEND_API_BASE_URL` in the frontend environment if the agent runs on another host/port.

## CLI: finance pipeline

From `src/agent` (with `venv` activated):

| Command | Purpose |
|--------|---------|
| `ingest-direct` | Ingest company-facts JSON |
| `ingest-edgar-local` | Ingest local `tools/data/EDGAR_*.htm` |
| `ask-direct` / `ask-multi` | Q&A over `document_id`(s) |
| `ask-multi --group NAME` | Q&A using `tools/data/document_groups.json` |
| `list-accessions` | List accessions from company-facts file |

Example:

```bash
cd src/agent
venv\Scripts\python scripts\run_sec_finance_pipeline.py ask-multi --document-ids 9002,9201 --question "..." --top-k 8
```

Full usage: see the docstring at the top of `scripts/run_sec_finance_pipeline.py`.

## Configuration notes

- **`EMBEDDING_DIMENSION`**: must match the Qdrant collection vector size (e.g. 1024 for DashScope `text-embedding-v3` if that is what you use).
- **`OPENSEARCH_SPARSE_SEARCH_SCOPE`**: defaults to `finance`; use `all` to query finance + default sparse index (see `env.example`).
- **Document groups**: JSON map of group name → list of `document_id` for CLI and `GET /agent/api/documents/groups`.

## Security

- **Never commit** `src/agent/.env` or API keys.
- Rotate keys if they were ever pushed; use `.gitignore` (already excludes common secret paths — verify before `git push`).

## Extending to other domains

The codebase is **finance-first**. To add another vertical (e.g. education), extend retrieval fields, sparse query profiles, and index routing — see module docstrings in `tools/retrieval_fields.py` and `tools/retrieval_backends/sparse_query_profiles.py`.

## License

Add a `LICENSE` file when you publish (e.g. MIT). This README does not grant any rights by itself.
