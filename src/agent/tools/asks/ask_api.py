"""问答 API，路由层仅负责参数校验与响应封装。"""

import json
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from core.config import config
from tools.chinese_converter import get_converter
from tools.evaluation_pipeline import run_pending_evaluations
from tools.finance.product_surface import get_finance_product_spec
from tools.llamaindex_retrieval import retrieval_service
from tools.llm import get_llm
from tools.rag_service import answer_question, stream_answer_events

router = APIRouter(prefix="/api/ask", tags=["Ask Generation"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    document_ids: List[int] = Field(..., min_items=1, max_items=50)
    top_k: int = Field(default=10, ge=1, le=50)
    detail_level: str = Field(default="detailed", pattern="^(brief|detailed|comprehensive)$")
    report_locale: Optional[str] = Field(
        default=None,
        description="产品层报告语言：zh / en / auto（按问题推断，默认 auto）",
    )
    include_pipeline_trace: bool = Field(
        default=False,
        description="为 true 时返回 pipeline_trace：检索计数、retrieval_sparse_hits（sparse 各阶段 node_id）、"
        "Bocha rerank、Langfuse 状态等",
    )
    include_full_retrieval_debug: bool = Field(
        default=False,
        description="为 true 时在 pipeline_trace 中附带 retrieval_debug 的键列表（不含大块正文）",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources_used: int
    citation_count: int = Field(default=0, description="内部检索生成的引用条数（不返回引用正文）")
    limitations: Optional[str] = None
    trace_id: Optional[str] = None
    latency_ms: Optional[float] = None
    pipeline_trace: Optional[Dict[str, Any]] = None
    vertical_scenario: Optional[Dict[str, Any]] = None
    external_evaluation: Optional[Dict[str, Any]] = None
    evidence_ui: Optional[Dict[str, Any]] = None
    report_locale: Optional[str] = Field(default=None, description="解析后的报告语言 zh 或 en")


class ParseDocumentsRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class DocumentRequirements(BaseModel):
    titles: Optional[List[str]] = Field(default=[])
    authors: Optional[List[str]] = Field(default=[])
    topics: Optional[List[str]] = Field(default=[])
    document_type: Optional[str] = Field(default="both")


class ParseDocumentsResponse(BaseModel):
    success: bool
    document_requirements: DocumentRequirements
    confidence: float = Field(..., ge=0.0, le=1.0)


class DocumentVectorSearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    document_ids: List[int] = Field(..., min_items=1, max_items=50)
    top_k: int = Field(default=10, ge=1, le=20)


class DocumentVectorSearchResult(BaseModel):
    document_id: int
    title: Optional[str] = None
    authors: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0)
    top_chunks: List[Dict[str, Any]] = Field(default=[])


class DocumentVectorSearchResponse(BaseModel):
    success: bool
    documents: List[DocumentVectorSearchResult]
    total_searched: int
    total_found: int


class ConvertChineseRequest(BaseModel):
    text: str
    mode: str = Field(default="s2t")


class ConvertChineseResponse(BaseModel):
    success: bool
    converted: str
    original: str


@router.post("/generate", response_model=AskResponse)
async def generate_answer(request: AskRequest):
    try:
        result = await answer_question(
            question=request.question,
            document_ids=request.document_ids,
            detail_level=request.detail_level,
            top_k=request.top_k,
            include_pipeline_trace=request.include_pipeline_trace,
            include_full_retrieval_debug=request.include_full_retrieval_debug,
            report_locale=request.report_locale,
        )
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources_used=result["sources_used"],
            citation_count=int(result.get("citation_count") or 0),
            limitations=result.get("limitations"),
            trace_id=result.get("trace_id"),
            latency_ms=result.get("latency_ms"),
            pipeline_trace=result.get("pipeline_trace"),
            vertical_scenario=result.get("vertical_scenario"),
            external_evaluation=result.get("external_evaluation"),
            evidence_ui=result.get("evidence_ui"),
            report_locale=result.get("report_locale"),
        )
    except Exception as exc:
        logger.exception("[AskAPI] generate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate/stream")
async def generate_answer_stream(request: AskRequest):
    try:
        return StreamingResponse(
            stream_answer_events(
                question=request.question,
                document_ids=request.document_ids,
                detail_level=request.detail_level,
                top_k=request.top_k,
                report_locale=request.report_locale,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as exc:
        logger.exception("[AskAPI] stream failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/product-spec")
async def finance_product_spec():
    return get_finance_product_spec()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "ask-generation",
        "langgraph_planner_enabled": config.enable_langgraph_planner,
        "ragas_enabled": config.ragas_enabled,
    }


@router.post("/parse-documents", response_model=ParseDocumentsResponse)
async def parse_documents(request: ParseDocumentsRequest):
    prompt = f"""分析用户问题，提取文档筛选条件，仅返回 JSON。
问题：{request.question}

格式：
{{
  "titles": [],
  "authors": [],
  "topics": [],
  "document_type": "both",
  "confidence": 0.5
}}"""
    try:
        llm = get_llm(temperature=0.0)
        response = await llm.ainvoke(
            [
                SystemMessage(content="你是文档检索过滤条件提取器，只返回 JSON。"),
                HumanMessage(content=prompt),
            ]
        )
        text = response.content if hasattr(response, "content") else str(response)
        try:
            parsed = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
        return ParseDocumentsResponse(
            success=True,
            document_requirements=DocumentRequirements(
                titles=parsed.get("titles", []) or [],
                authors=parsed.get("authors", []) or [],
                topics=parsed.get("topics", []) or [],
                document_type=parsed.get("document_type", "both") or "both",
            ),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
        )
    except Exception:
        return ParseDocumentsResponse(
            success=True,
            document_requirements=DocumentRequirements(),
            confidence=0.0,
        )


@router.post("/search-documents-vector", response_model=DocumentVectorSearchResponse)
async def search_documents_vector(request: DocumentVectorSearchRequest):
    try:
        result = await retrieval_service.retrieve(
            query=request.question,
            document_ids=request.document_ids,
        )
        grouped: dict[int, list[dict[str, Any]]] = {}
        for node in result["nodes"]:
            document_id = int(node["document_id"])
            grouped.setdefault(document_id, []).append(node)

        documents = []
        for document_id, nodes in grouped.items():
            ranked = sorted(nodes, key=lambda item: item.get("score", 0.0), reverse=True)
            documents.append(
                DocumentVectorSearchResult(
                    document_id=document_id,
                    title=ranked[0].get("title") or f"文档 {document_id}",
                    authors=None,
                    score=min(1.0, float(ranked[0].get("score") or 0.0)),
                    top_chunks=[
                        {
                            "section_number": int(item.get("order_index") or 0) + 1,
                            "text_preview": (item.get("text") or "")[:200],
                            "score": float(item.get("score") or 0.0),
                            "node_id": item.get("node_id"),
                        }
                        for item in ranked[:3]
                    ],
                )
            )

        documents.sort(key=lambda item: item.score, reverse=True)
        documents = documents[: request.top_k]
        return DocumentVectorSearchResponse(
            success=True,
            documents=documents,
            total_searched=len(request.document_ids),
            total_found=len(grouped),
        )
    except Exception as exc:
        logger.exception("[AskAPI] search-documents-vector failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/evaluate/pending")
async def evaluate_pending_traces():
    try:
        return await run_pending_evaluations()
    except Exception as exc:
        logger.exception("[AskAPI] evaluate/pending failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/utils/convert-chinese", response_model=ConvertChineseResponse)
async def convert_chinese_text(request: ConvertChineseRequest):
    try:
        if request.mode not in ["s2t", "t2s"]:
            raise HTTPException(status_code=400, detail="mode 必须是 's2t' 或 't2s'")
        converted = get_converter().convert(request.text, request.mode)
        return ConvertChineseResponse(success=True, converted=converted, original=request.text)
    except Exception:
        return ConvertChineseResponse(success=False, converted=request.text, original=request.text)
