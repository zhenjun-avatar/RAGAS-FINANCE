"""FastAPI entrypoint for the rebuilt node-centric RAG API."""

import json
import logging

from core.config import config
from tools.runtime_logging import configure_runtime_logging

configure_runtime_logging()

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tools.bocha_reranker import reranker
from tools.finance.financial_facts_repository import query_observations
from tools.ingestion_service import process_document, reindex_document_vectors
from tools.langfuse_tracing import tracer
from tools.document_groups import default_document_groups_path, load_document_groups
from tools.node_repository import ensure_schema, list_available_document_ids, list_document_catalog
from tools.report_store import get_report, list_reports, save_ask_report

logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Node-centric RAG API",
    description="Node-based ingestion, LlamaIndex retrieval, Bocha rerank, Langfuse tracing",
    version="2.0.0",
)

agent_router = APIRouter(prefix="/agent")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in config.cors_allowed_origins.split(",")
    if origin.strip()
]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With", "x-user-id"],
    expose_headers=["*"],
)

from tools.asks.ask_api import router as ask_router

agent_router.include_router(ask_router)


@app.on_event("startup")
async def startup_event():
    try:
        await ensure_schema()
    except Exception as exc:
        logger.error(
            "PostgreSQL connection failed (host=%s port=%s db=%s). "
            "Start the server and ensure DATABASE_URL or DB_* in .env is correct. Underlying: %s",
            config.db_host,
            config.db_port,
            config.db_name,
            exc,
        )
        raise RuntimeError(
            f"PostgreSQL unreachable at {config.db_host}:{config.db_port}/{config.db_name}. "
            "Fix DB connectivity, then restart the API. "
            "Local deps: from repo root run  docker compose -f docker-compose.rag.yml up -d "
            "(default host port 5433; if you use a local Postgres on 5432, set DB_PORT=5433 and DATABASE_URL "
            "to match compose, not 5432.)"
        ) from exc
    logger.info(
        "Pipeline: bocha_rerank=%s langfuse=%s",
        reranker.describe_config(),
        tracer.diagnostics(),
    )


class DiagramRequest(BaseModel):
    content: str
    user_prompt: str
    title: str = ""


class DocumentProcessRequest(BaseModel):
    document_id: int
    file_path: str
    file_type: str


class RevectorizeRequest(BaseModel):
    document_id: int


class ReportSaveRequest(BaseModel):
    request: dict
    response: dict
    source: str = "frontend"


@agent_router.get("/")
async def root():
    return {
        "message": "Node-centric RAG API",
        "version": "2.0.0",
        "status": "running",
        "scope": ["ingestion", "retrieval", "rerank", "observability", "evaluation"],
    }


@agent_router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "allowed_origins": ALLOWED_ORIGINS,
        "langfuse_enabled": config.langfuse_enabled,
        "langgraph_planner_enabled": config.enable_langgraph_planner,
        "bocha_rerank": reranker.describe_config(),
        "langfuse_client": tracer.diagnostics(),
    }


@agent_router.get("/api/observability/pipeline")
async def pipeline_observability():
    """各步骤配置与 Langfuse 客户端状态（不含密钥）；用于确认 rerank / 追踪是否就绪。"""
    return {
        "bocha_rerank": reranker.describe_config(),
        "langfuse": tracer.diagnostics(),
        "langfuse_env_flag": config.langfuse_enabled,
        "langgraph_planner": config.enable_langgraph_planner,
    }


@agent_router.get("/models")
async def list_models():
    return {
        "models": [
            "deepseek/deepseek-chat",
            "deepseek/deepseek-coder",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "qwen/qwen-turbo",
        ],
        "default": config.default_model,
    }


@agent_router.post("/api/document-diagram/generate")
async def generate_diagram_endpoint(request: DiagramRequest):
    try:
        from tools.document_structure_analyzer import generate_document_diagram

        return await generate_document_diagram(
            content=request.content,
            user_prompt=request.user_prompt,
            title=request.title,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.get("/api/finance/observations")
async def list_finance_observations(
    cik: int | None = None,
    document_id: int | None = None,
    taxonomy: str | None = None,
    metric_key: str | None = None,
    form: str | None = None,
    limit: int = 200,
):
    """SQL-first filter on ingested SEC company-facts rows (due diligence)."""
    try:
        rows = await query_observations(
            cik=cik,
            document_id=document_id,
            taxonomy=taxonomy,
            metric_key=metric_key,
            form=form,
            limit=min(max(limit, 1), 2000),
        )
        return {"count": len(rows), "observations": rows}
    except Exception as exc:
        logger.exception("[API] finance observations failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.post("/api/documents/process")
async def process_document_api(request: DocumentProcessRequest):
    try:
        return await process_document(
            file_path=request.file_path,
            file_type=request.file_type,
            document_id=request.document_id,
        )
    except Exception as exc:
        logger.exception("[API] process document failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.post("/api/documents/revectorize")
async def revectorize_document_api(request: RevectorizeRequest):
    try:
        return await reindex_document_vectors(request.document_id)
    except Exception as exc:
        logger.exception("[API] revectorize failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.get("/api/documents/ids")
async def list_document_ids_api(limit: int = 500):
    try:
        ids = await list_available_document_ids(limit=min(max(limit, 1), 5000))
        return {"count": len(ids), "document_ids": ids}
    except Exception as exc:
        logger.exception("[API] list document ids failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.get("/api/documents/catalog")
async def list_document_catalog_api(limit: int = 500):
    try:
        items = await list_document_catalog(limit=min(max(limit, 1), 5000))
        return {"count": len(items), "items": items}
    except Exception as exc:
        logger.exception("[API] list document catalog failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.get("/api/documents/groups")
async def list_document_groups_api():
    """Return named document id groups from ``tools/data/document_groups.json`` (CLI-compatible)."""
    path = default_document_groups_path()
    if not path.is_file():
        return {"groups": {}, "path": str(path), "missing": True}
    try:
        groups = load_document_groups(path)
        return {"groups": groups, "path": str(path), "missing": False}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.exception("[API] load document groups failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.post("/api/report/save")
async def save_report_api(request: ReportSaveRequest):
    try:
        out = save_ask_report(
            request_payload=request.request if isinstance(request.request, dict) else {},
            response_payload=request.response if isinstance(request.response, dict) else {},
            source=str(request.source or "frontend"),
        )
        return {"ok": True, **out}
    except Exception as exc:
        logger.exception("[API] save report failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.get("/api/report/list")
async def list_reports_api(limit: int = 50):
    try:
        items = list_reports(limit=min(max(limit, 1), 500))
        return {"count": len(items), "items": items}
    except Exception as exc:
        logger.exception("[API] list reports failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@agent_router.get("/api/report/{trace_id}")
async def get_report_api(trace_id: str):
    try:
        item = get_report(trace_id)
        if item is None:
            raise HTTPException(status_code=404, detail="report not found")
        return item
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[API] get report failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.include_router(agent_router)


def create_app() -> FastAPI:
    return app


def run_server(host: str = None, port: int = None, debug: bool = None):
    import uvicorn

    host = host or config.host
    port = port or config.port
    debug = debug if debug is not None else config.debug
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=debug,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
