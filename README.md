# RAGAS-FINANCE — Financial Disclosure RAG

Node-centric RAG for **SEC-style filings** (HTML + optional company-facts alignment): **hybrid retrieval** (dense + sparse), optional **reranking**, **Langfuse** traces, **Next.js** UI.

## Architecture (ask path)

1. **Routing** — Finance-aware **evidence plan** (narrative targets, filing scope, period hints).
2. **Retrieval** — Hybrid search on Postgres-backed nodes: **Qdrant** (dense) + **Postgres full-text** or **OpenSearch** (sparse, finance profile). Narrative queries: **section tree** hits → **leaf descendants** → fusion (e.g. RRF) → optional **rerank** (e.g. Bocha).
3. **Context** — **Sibling expansion** (bounded seeds/limit/decay) + `**CONTEXT_CHAR_BUDGET`** and optional **title match** for `narrative_targets`; not only fixed `top_k`.
4. **Answer** — LLM over retrieved context; **Langfuse** optional.
5. **Eval (optional)** — RAGAS jobs → `POST /agent/api/ask/evaluate/pending` (batch size `RAGAS_BATCH_SIZE`) → scores to Langfuse.

**Ingest** — Multi-level **section tree** from `section_path` (leaves = chunks); vectors + sparse index; metadata e.g. `finance_accns`, form, periods.

## Evaluation (RAGAS → Langfuse)


|             |                                                                                                                                                                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset** | `[apple_narrative_questions_100.json](src/agent/tools/data/apple_narrative_questions_100.json)` — 100 narrative questions (Apple CIK 320193), topic groups A–J. Set `**document_id_range`** to match your ingest `document_id`s. |
| **Metrics** | `faithfulness`, `context_precision` — require `RAGAS_ENABLED` + `LANGFUSE_*` in `.env`.                                                                                                                                          |


**Sample run (n = 100)** — snapshot only; varies by corpus and config.

| Dimension | Example snapshot (n = 100) |
| --------- | -------------------------- |
| Quality (RAGAS) | `context_precision`, `faithfulness` (configuration- and corpus-dependent) |
| Latency | ~10.9 s average end-to-end per question |
| Cost | ~1.9K average tokens per question |

| Metric              | n   | Avg  | ×0  | ×1  |
| ------------------- | --- | ---- | --- | --- |
| `context_precision` | 100 | 0.61 | 15  | 38  |
| `faithfulness`      | 100 | 0.90 | 4   | 80  |

Scores depend on corpus, retrieval, prompts, and `top_k` / context budget.

**Scripts** (API up, `cd src/agent`):  
`python scripts/run_mixed_narrative_questions_parallel.py --questions tools/data/apple_narrative_questions_100.json`  
`python scripts/run_evaluate_pending_parallel.py`

## Features

- Ingest local **EDGAR `.htm`**, align accession/form/dates via company-facts JSON where used.
- **Hybrid** dense + sparse; finance-tuned sparse profile.
- **FastAPI** + document **groups**; **Next.js** frontend.

## Layout

```
rag-testing/
├── docker-compose.rag.yml
├── requirements.txt
├── src/agent/          # FastAPI, ingest, RAG, scripts/, tools/, env.example
└── src/frontend/       # Next.js
```

## Prerequisites

- Docker (Postgres, Qdrant; OpenSearch if `SPARSE_BACKEND=opensearch`)
- Python 3.11+, Node 18+
- Embedding + LLM keys — see `src/agent/env.example`

## Quick start

```bash
docker compose -f docker-compose.rag.yml up -d
cp src/agent/env.example src/agent/.env   # DATABASE_URL, Qdrant, keys, SPARSE_BACKEND, …

python -m venv src/agent/venv
src/agent/venv/Scripts/pip install -r requirements.txt   # Windows
# src/agent/venv/bin/pip install -r requirements.txt     # Unix

cd src/agent && venv/Scripts/python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

```bash
cd src/frontend && npm install && npm run dev
```

## Document ingest (step-by-step)

All ingest commands run from `**src/agent**` with a virtualenv active and `**src/agent/.env**` configured (`DATABASE_URL`, Qdrant, `EMBEDDING_PROVIDER` + keys, `SPARSE_BACKEND`, OpenSearch fields if applicable). Start Postgres and Qdrant first (`docker compose -f docker-compose.rag.yml up -d` from the repo root).

**What ingest does** — Parses EDGAR-style HTML into a **section tree** (leaves = chunks), writes rows to Postgres, upserts **dense** vectors (Qdrant) and **sparse** index (Postgres full-text or OpenSearch, depending on config). Optional **companyfacts** JSON aligns accession, form, filing date, and entity metadata.

### 1. EDGAR filing HTML (narrative RAG)

Each filing gets its own `**document_id`**, assigned in order from `**--document-id-start`**. Plan ranges so facts + filings do not overlap IDs.

**A. Local `.htm` files** (recommended for repeatable runs) — Files under `--data-dir` matching `--edgar-glob` are ingested one file per `document_id`. Pass `**--companyfacts-json`** so accession-level **form** / **filed** / **entityName** align when filenames embed CIK and accession.

```bash
# Windows — example: Apple CIK 320193, 70 files → document_id 9801–9870
venv\Scripts\python scripts\run_sec_finance_pipeline.py ingest-edgar-local --document-id-start 9801 --data-dir tools\data --edgar-glob "EDGAR_320193_*.htm" --companyfacts-json tools\data\CIK0000320193.json
```

```bash
# Unix
venv/bin/python scripts/run_sec_finance_pipeline.py ingest-edgar-local --document-id-start 9801 --data-dir tools/data --edgar-glob 'EDGAR_320193_*.htm' --companyfacts-json tools/data/CIK0000320193.json
```

Optional overrides: `--form`, `--filed` (YYYY-MM-DD), `--entity-name` when not inferable from JSON.

**B. Download from SEC** — Uses accessions from the companyfacts JSON (or `--accession` filters). Respects `**--edgar-delay`** between requests. `**download-edgar`** saves `.htm` only; `**ingest-edgar**` downloads and ingests (same flags: `--document-id-start`, `--max-filings`, `--download-dir`).

```bash
venv\Scripts\python scripts\run_sec_finance_pipeline.py list-accessions --json-path tools\data\CIK0000320193.json
venv\Scripts\python scripts\run_sec_finance_pipeline.py ingest-edgar --document-id-start 9100 --max-filings 5 --json-path tools\data\CIK0000320193.json
```

### 2. After config or backend changes

If you change `**SPARSE_BACKEND**` (e.g. to OpenSearch) or embedding model / dimension, **re-ingest** affected documents or re-run vector indexing as appropriate. For vector reindex only:

```bash
venv\Scripts\python scripts\run_sec_finance_pipeline.py reindex-vectors --document-id <id>
```

Full flag reference: docstring at the top of `[scripts/run_sec_finance_pipeline.py](src/agent/scripts/run_sec_finance_pipeline.py)`.

## CLI (`src/agent`)


| Command                    | Purpose                                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `ingest-edgar-local`       | Local `EDGAR_*.htm` → nodes + indexes (`--document-id-start`, `--edgar-glob`, `--companyfacts-json`) |
| `ingest-direct`            | Company-facts JSON into DB (metadata / pipeline use)                                                 |
| `ask-direct` / `ask-multi` | Q&A over `document_id`(s)                                                                            |
| `ask-multi --group`        | Use `tools/data/document_groups.json`                                                                |


```bash
venv/Scripts/python scripts/run_sec_finance_pipeline.py ask-multi --document-ids 9801 --question "What does the MD&A of Apple's fiscal year 2024 10-K say about liquidity and capital resources? Cite the specific filing narrative and any cash, marketable securities, or debt figures that appear in the retrieved passages." --top-k 8
```

More: docstring in `scripts/run_sec_finance_pipeline.py`.

## Config (see `core/config.py`, `env.example`)

- `EMBEDDING_DIMENSION` — must match Qdrant collection size.
- `CONTEXT_CHAR_BUDGET`, `CONTEXT_SIBLING_*`, `SECTION_TREE_SEARCH_DEPTH` — context assembly.
- `OPENSEARCH_SPARSE_SEARCH_SCOPE` — if using OpenSearch sparse.

## Security

Do not commit `.env` or keys. Verify `.gitignore` before pushing.

## Extending

Finance-oriented retrieval fields and sparse profiles: `tools/retrieval_fields.py`, `tools/retrieval_backends/sparse_query_profiles.py`.

## License

MIT