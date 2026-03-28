"""Document ingestion pipeline for node-centric RAG storage."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from core.config import config
from .chunk_segmenter import ChunkPayload, build_retrieval_chunks
from .document_parser import parse_document_content
from .finance.financial_facts_repository import replace_sec_observations
from .finance.sec_company_facts import (
    batch_lines_for_nodes,
    batch_row_groups_for_nodes,
    build_chunk_filter_metadata,
    flatten_sec_company_facts,
    is_sec_company_facts_payload,
)
from .llm import get_llm
from .node_repository import (
    NodeRecord,
    ensure_schema,
    finish_ingest_run,
    list_document_nodes,
    mark_nodes_vectorized,
    replace_document_nodes,
    start_ingest_run,
    upsert_document,
)
from .retrieval_fields import FIELD_METADATA_KEY, build_retrieval_fields
from .retrieval_backends.factory import get_dense_backend, get_sparse_backend
from .vectorizer import generate_embeddings_batch


def _has_summary_model_configured() -> bool:
    return bool(config.deepseek_api_key or config.qwen_api_key or config.openai_api_key)


_NODE_METADATA_PASSTHROUGH_KEYS: tuple[str, ...] = (
    # routing / domain
    "domain",
    "source_kind",
    # SEC / finance identity
    "cik",
    "entity_name",
    "sec_accession",
    "sec_filing_date",
    "primary_document",
    # retrieval filters
    "finance_forms",
    "finance_form_base",
    "finance_accns",
    "finance_metric_exact_keys",
    "finance_period_years",
    "finance_period_end_dates",
    # parser artifacts that are useful for diagnostics/citations
    "unstructured_json",
)


def _normalize_finance_form(form: str | None) -> str | None:
    text = str(form or "").strip().upper().replace(" ", "")
    if not text:
        return None
    text = text.replace(".", "-")
    text = text.replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K")
    if text.endswith("/A"):
        text = text[:-2]
    return text


def _detect_summary_language(text: str, metadata: dict[str, Any]) -> str:
    hinted = str(metadata.get("language") or "").strip().lower()
    if hinted.startswith("en"):
        return "en"
    if hinted.startswith("zh"):
        return "zh"
    sample = (text or "")[:4000]
    if not sample:
        return "zh"
    ascii_letters = sum(1 for ch in sample if ("a" <= ch.lower() <= "z"))
    cjk_chars = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    return "en" if ascii_letters > (cjk_chars * 2 + 40) else "zh"


def _copy_metadata_value(value: Any) -> Any:
    """Shallow-safe copy for scalar/list metadata values used by node metadata."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [item for item in value if item is not None]
    if isinstance(value, tuple):
        return [item for item in value if item is not None]
    return None


def _base_node_metadata(document_metadata: dict[str, Any]) -> dict[str, Any]:
    base = {
        key: value
        for key, value in {
            "source_file_type": document_metadata.get("file_type"),
            "source_file_name": document_metadata.get("file_name"),
            "source_parser": document_metadata.get("parser"),
            "source_page_count": document_metadata.get("page_count"),
            "document_char_count": document_metadata.get("char_count"),
        }.items()
        if value is not None
    }
    # Preserve ingestion-level metadata required by finance retrieval filters.
    for key in _NODE_METADATA_PASSTHROUGH_KEYS:
        copied = _copy_metadata_value(document_metadata.get(key))
        if copied is not None:
            base[key] = copied
    return base


def _build_summary_text(group_index: int, chunks: list[ChunkPayload]) -> str:
    parts: list[str] = []
    for chunk in chunks[:3]:
        label = chunk.metadata.get("section_title") or chunk.title
        preview = chunk.text[:320].strip()
        if label:
            parts.append(f"{label}\n{preview}")
        else:
            parts.append(preview)
    body = "\n\n".join(part for part in parts if part).strip()
    return f"Summary Group {group_index + 1}\n\n{body[:1200]}"


def _build_root_fallback_summary(summary_texts: list[str], full_text: str) -> str:
    body = "\n\n".join(text.strip() for text in summary_texts[:4] if text.strip())
    if body:
        return body[:1800]
    return full_text[:1800]


def _normalize_summary_output(text: str, fallback: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
    if cleaned.lower().startswith("summary:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned or fallback


def _format_chunk_context(chunk: ChunkPayload, index: int) -> str:
    labels: list[str] = [f"Chunk {index}"]
    if chunk.metadata.get("section_title"):
        labels.append(f"section={chunk.metadata['section_title']}")
    if chunk.metadata.get("page_start") is not None:
        page_start = chunk.metadata["page_start"]
        page_end = chunk.metadata.get("page_end", page_start)
        labels.append(f"pages={page_start}-{page_end}")
    header = " | ".join(labels)
    return f"{header}\n{chunk.text[:900].strip()}"


async def _summarize_chunk_group(group_index: int, chunks: list[ChunkPayload]) -> tuple[str, str]:
    fallback = _build_summary_text(group_index, chunks)
    if not _has_summary_model_configured():
        return fallback, "extractive"

    context = "\n\n".join(_format_chunk_context(chunk, idx) for idx, chunk in enumerate(chunks, start=1))
    lang = _detect_summary_language(context, chunks[0].metadata if chunks else {})
    llm = get_llm(model_name=config.default_model, temperature=0.0)
    try:
        if lang == "en":
            system_prompt = (
                "You are a retrieval summarizer. Summarize adjacent document chunks for vector and BM25 retrieval. "
                "Preserve key entities, conclusions, terminology, and section intent. No fabrication."
            )
            human_prompt = (
                f"Write a 120-220 word summary for chunk group {group_index + 1}.\n\n"
                f"{context[:3200]}"
            )
        else:
            system_prompt = (
                "你是检索层摘要器。请为一组相邻文档片段生成适合向量检索和 BM25 检索的摘要。"
                "保留主题、关键实体、结论、术语和章节语义。不要编造，不要输出列表或前缀。"
            )
            human_prompt = (
                f"请为第 {group_index + 1} 组片段生成 120 到 220 字摘要。\n\n"
                f"{context[:3200]}"
            )
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        summary_text = _normalize_summary_output(
            response.content if hasattr(response, "content") else str(response),
            fallback,
        )
        if len(summary_text) < 40:
            return fallback, "extractive"
        return summary_text[:800], "llm"
    except Exception as exc:
        logger.warning(
            "[Ingestion] Group summary fallback for document chunk group {}: {}",
            group_index,
            exc,
        )
        return fallback, "extractive"


async def _summarize_document(
    summary_texts: list[str],
    full_text: str,
    document_metadata: dict[str, Any],
) -> tuple[str, str]:
    fallback = _build_root_fallback_summary(summary_texts, full_text)
    if not _has_summary_model_configured():
        return fallback, "extractive"

    doc_title = document_metadata.get("file_name") or "document"
    context = "\n\n".join(
        f"Section {idx}\n{text[:700].strip()}"
        for idx, text in enumerate(summary_texts[:6], start=1)
        if text.strip()
    )
    lang = _detect_summary_language(f"{doc_title}\n{context}", document_metadata)
    llm = get_llm(model_name=config.default_model, temperature=0.0)
    try:
        if lang == "en":
            system_prompt = (
                "You are a document-level retrieval summarizer. Produce a parent-node summary for retrieval, "
                "covering topic, key entities, major claims, conclusions, and section structure. No fabrication."
            )
            human_prompt = (
                f"Document title: {doc_title}\n"
                "Write a 180-320 word summary for the top retrieval node.\n\n"
                f"{context[:3600]}"
            )
        else:
            system_prompt = (
                "你是检索层文档摘要器。请输出适合父节点召回的文档级摘要，"
                "覆盖主题、关键实体、主要论点、结论和章节结构线索。不要编造。"
            )
            human_prompt = (
                f"文档标题：{doc_title}\n"
                "请生成 180 到 320 字摘要，用于上层检索节点。\n\n"
                f"{context[:3600]}"
            )
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        summary_text = _normalize_summary_output(
            response.content if hasattr(response, "content") else str(response),
            fallback,
        )
        if len(summary_text) < 60:
            return fallback, "extractive"
        return summary_text[:1200], "llm"
    except Exception as exc:
        logger.warning("[Ingestion] Document summary fallback: {}", exc)
        return fallback, "extractive"


def _merge_chunk_metadata(document_metadata: dict[str, Any], chunk: ChunkPayload) -> dict[str, Any]:
    merged = _base_node_metadata(document_metadata)
    merged.update(chunk.metadata)
    return merged


def _merge_group_metadata(document_metadata: dict[str, Any], chunks: list[ChunkPayload]) -> dict[str, Any]:
    merged = _base_node_metadata(document_metadata)
    merged["group_size"] = len(chunks)

    page_starts = [int(chunk.metadata["page_start"]) for chunk in chunks if "page_start" in chunk.metadata]
    page_ends = [int(chunk.metadata["page_end"]) for chunk in chunks if "page_end" in chunk.metadata]
    section_titles = [
        str(chunk.metadata["section_title"])
        for chunk in chunks
        if chunk.metadata.get("section_title")
    ]
    heading_paths = [chunk.metadata.get("heading_path") for chunk in chunks if chunk.metadata.get("heading_path")]

    if page_starts:
        merged["page_start"] = min(page_starts)
        merged["page_end"] = max(page_ends or page_starts)
    if section_titles:
        merged["section_title"] = section_titles[0]
        merged["section_titles"] = list(dict.fromkeys(section_titles))[:4]
    if heading_paths:
        merged["heading_path"] = heading_paths[0]
        merged["section_depth"] = len(heading_paths[0])

    return merged


def _attach_retrieval_fields(
    *,
    node_type: str,
    level: int,
    title: str | None,
    text: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    out = dict(metadata or {})
    out[FIELD_METADATA_KEY] = build_retrieval_fields(
        node_type=node_type,
        level=level,
        title=title,
        text=text,
        metadata=out,
    )
    return out


def _node_index_record(node: NodeRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(node, NodeRecord):
        metadata = dict(node.metadata or {})
        retrieval_fields = metadata.get(FIELD_METADATA_KEY)
        if not isinstance(retrieval_fields, dict):
            metadata = _attach_retrieval_fields(
                node_type=node.node_type,
                level=node.level,
                title=node.title,
                text=node.text,
                metadata=metadata,
            )
            retrieval_fields = metadata.get(FIELD_METADATA_KEY) or {}
        return {
            "node_id": node.node_id,
            "document_id": node.document_id,
            "ingest_run_id": node.ingest_run_id,
            "parent_id": node.parent_id,
            "node_type": node.node_type,
            "level": node.level,
            "order_index": node.order_index,
            "title": node.title,
            "text": node.text,
            "metadata": metadata,
            **retrieval_fields,
        }
    metadata = dict(node.get("metadata") or {})
    retrieval_fields = metadata.get(FIELD_METADATA_KEY)
    if not isinstance(retrieval_fields, dict):
        metadata = _attach_retrieval_fields(
            node_type=node["node_type"],
            level=node["level"],
            title=node.get("title"),
            text=node["text"],
            metadata=metadata,
        )
        retrieval_fields = metadata.get(FIELD_METADATA_KEY) or {}
    return {
        "node_id": node["node_id"],
        "document_id": node["document_id"],
        "ingest_run_id": node.get("ingest_run_id"),
        "parent_id": node.get("parent_id"),
        "node_type": node["node_type"],
        "level": node["level"],
        "order_index": node["order_index"],
        "title": node.get("title"),
        "text": node["text"],
        "metadata": metadata,
        **retrieval_fields,
    }


async def _build_nodes(document_id: int, ingest_run_id: str, text: str, metadata: dict[str, Any]) -> list[NodeRecord]:
    chunks = build_retrieval_chunks(text)
    summary_group_size = max(2, config.hierarchical_group_size)
    nodes: list[NodeRecord] = []
    grouped_chunks = [
        chunks[group_index : group_index + summary_group_size]
        for group_index in range(0, len(chunks), summary_group_size)
        if chunks[group_index : group_index + summary_group_size]
    ]
    group_summaries = await asyncio.gather(
        *[
            _summarize_chunk_group(group_index, group_chunks)
            for group_index, group_chunks in enumerate(grouped_chunks)
        ]
    ) if grouped_chunks else []

    for summary_order, group_chunks in enumerate(grouped_chunks):
        summary_id = str(uuid.uuid4())
        summary_text, summary_method = group_summaries[summary_order]
        summary_metadata = _merge_group_metadata(metadata, group_chunks)
        summary_metadata["summary_method"] = summary_method
        nodes.append(
            NodeRecord(
                node_id=summary_id,
                document_id=document_id,
                ingest_run_id=ingest_run_id,
                node_type="summary",
                level=1,
                order_index=summary_order,
                title=group_chunks[0].metadata.get("section_title") or f"Summary {summary_order + 1}",
                text=summary_text,
                metadata=_attach_retrieval_fields(
                    node_type="summary",
                    level=1,
                    title=group_chunks[0].metadata.get("section_title") or f"Summary {summary_order + 1}",
                    text=summary_text,
                    metadata=summary_metadata,
                ),
            )
        )
        for inner_index, chunk in enumerate(group_chunks):
            absolute_index = summary_order * summary_group_size + inner_index
            chunk_title = chunk.title or f"Chunk {absolute_index + 1}"
            chunk_text = chunk.text
            nodes.append(
                NodeRecord(
                    node_id=str(uuid.uuid4()),
                    document_id=document_id,
                    ingest_run_id=ingest_run_id,
                    parent_id=summary_id,
                    node_type="chunk",
                    level=0,
                    order_index=absolute_index,
                    title=chunk_title,
                    text=chunk_text,
                    metadata=_attach_retrieval_fields(
                        node_type="chunk",
                        level=0,
                        title=chunk_title,
                        text=chunk_text,
                        metadata=_merge_chunk_metadata(metadata, chunk),
                    ),
                )
            )

    if nodes:
        root_id = str(uuid.uuid4())
        summary_texts = [item.text for item in nodes if item.level == 1]
        root_text, root_method = await _summarize_document(summary_texts, text, metadata)
        root_metadata = _base_node_metadata(metadata)
        root_metadata["summary_method"] = root_method
        root = NodeRecord(
            node_id=root_id,
            document_id=document_id,
            ingest_run_id=ingest_run_id,
            node_type="document_summary",
            level=2,
            order_index=0,
            title="Document Summary",
            text=root_text or text[:2000],
            metadata=_attach_retrieval_fields(
                node_type="document_summary",
                level=2,
                title="Document Summary",
                text=root_text or text[:2000],
                metadata=root_metadata,
            ),
        )
        remapped: list[NodeRecord] = [root]
        for item in nodes:
            if item.level == 1:
                item.parent_id = root_id
            remapped.append(item)
        return remapped
    return nodes


async def _write_dense_sparse_for_document(
    document_id: int,
    *,
    node_count: int,
    index_records: list[dict[str, Any]],
    vector_payloads: list[dict[str, Any]],
) -> str | None:
    """Upsert Qdrant + OpenSearch. Avoid clearing Qdrant when we have DB nodes but zero embeddings."""
    dense_backend = get_dense_backend()
    sparse_backend = get_sparse_backend()
    if vector_payloads:
        dense_backend.replace_document_nodes(document_id, vector_payloads)
    elif node_count > 0:
        logger.error(
            "[Ingestion] document_id={}: {} nodes in Postgres but 0 embeddings — "
            "skipping Qdrant replace (prevents emptying vectors). Fix EMBEDDING_PROVIDER/API, "
            "then run: python scripts/run_sec_finance_pipeline.py reindex-vectors --document-id {}",
            document_id,
            node_count,
            document_id,
        )
    else:
        dense_backend.replace_document_nodes(document_id, [])
    await sparse_backend.replace_document_nodes(document_id, index_records)
    if node_count > 0 and not vector_payloads:
        return (
            f"No embeddings for document {document_id} ({node_count} nodes). "
            "Check EMBEDDING_PROVIDER / QWEN_API_KEY / OPENAI_API_KEY, then re-ingest or reindex vectors."
        )
    return None


async def _ingest_parsed_text_pipeline(
    document_id: int,
    ingest_run_id: str,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    nodes = await _build_nodes(document_id, ingest_run_id, text, metadata)
    await replace_document_nodes(document_id, ingest_run_id, nodes)
    index_records = [_node_index_record(item) for item in nodes]
    node_texts = [item.text for item in nodes]
    embeddings = await generate_embeddings_batch(node_texts)
    vector_payloads: list[dict[str, Any]] = []
    vectorized_ids: list[str] = []
    for item, embedding, record in zip(nodes, embeddings, index_records):
        if not embedding:
            continue
        vectorized_ids.append(item.node_id)
        vector_payloads.append({**record, "vector": embedding})
    ingest_err = await _write_dense_sparse_for_document(
        document_id,
        node_count=len(nodes),
        index_records=index_records,
        vector_payloads=vector_payloads,
    )
    if vector_payloads:
        await mark_nodes_vectorized(vectorized_ids)
    await finish_ingest_run(ingest_run_id, error=ingest_err)
    return {
        "success": ingest_err is None,
        "document_id": document_id,
        "ingest_run_id": ingest_run_id,
        "node_count": len(nodes),
        "vectorized_count": len(vector_payloads),
        "word_count": len(text),
        "metadata": metadata,
        "error": ingest_err,
    }


def _should_ingest_sec_company_facts(file_path: str, file_type: str) -> bool:
    path = Path(file_path)
    if path.suffix.lower() != ".json":
        return False
    ft = (file_type or "").lower().strip()
    if ft == "sec_companyfacts":
        return True
    if ft == "json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return is_sec_company_facts_payload(data)
    return False


async def _ingest_sec_company_facts_json(
    file_path: str,
    document_id: int,
    ingest_run_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Structured SQL (exact facts) + chunked nodes for BM25/vector (semantic)."""
    rows = flatten_sec_company_facts(data)
    await replace_sec_observations(document_id, ingest_run_id, rows)

    cik = data.get("cik")
    entity = data.get("entityName") or ""
    summary = (
        f"SEC EDGAR company facts (XBRL): {entity} | CIK {cik} | {len(rows)} observations. "
        "Use SQL/API for precise metric filters; chunks below are for semantic retrieval."
    )
    lines_per_chunk = 64
    line_chunks = batch_lines_for_nodes(rows, lines_per_chunk=lines_per_chunk)
    row_groups = batch_row_groups_for_nodes(rows, lines_per_chunk=lines_per_chunk)
    if not line_chunks and not rows:
        line_chunks = ["(no observations in facts payload)"]
        row_groups = [[]]

    base_meta: dict[str, Any] = {
        "domain": "finance",
        "source_kind": "sec_companyfacts",
        "cik": cik,
        "entity_name": entity,
        "source_file_name": os.path.basename(file_path),
        "pipeline_version": config.rag_pipeline_version,
    }
    root_id = str(uuid.uuid4())
    nodes: list[NodeRecord] = [
        NodeRecord(
            node_id=root_id,
            document_id=document_id,
            ingest_run_id=ingest_run_id,
            node_type="document_summary",
            level=2,
            order_index=0,
            title="Document Summary",
            text=summary,
            metadata=_attach_retrieval_fields(
                node_type="document_summary",
                level=2,
                title="Document Summary",
                text=summary,
                metadata=dict(base_meta),
            ),
        )
    ]
    for idx, chunk_text in enumerate(line_chunks):
        title = f"SEC facts {idx + 1}/{len(line_chunks)}"
        chunk_meta = dict(base_meta)
        if idx < len(row_groups):
            chunk_meta.update(build_chunk_filter_metadata(row_groups[idx]))
        nodes.append(
            NodeRecord(
                node_id=str(uuid.uuid4()),
                document_id=document_id,
                ingest_run_id=ingest_run_id,
                parent_id=root_id,
                node_type="chunk",
                level=0,
                order_index=idx,
                title=title,
                text=chunk_text,
                metadata=_attach_retrieval_fields(
                    node_type="chunk",
                    level=0,
                    title=title,
                    text=chunk_text,
                    metadata=chunk_meta,
                ),
            )
        )

    await replace_document_nodes(document_id, ingest_run_id, nodes)
    index_records = [_node_index_record(item) for item in nodes]
    node_texts = [item.text for item in nodes]
    embeddings = await generate_embeddings_batch(node_texts)
    vector_payloads: list[dict[str, Any]] = []
    vectorized_ids: list[str] = []
    for item, embedding, record in zip(nodes, embeddings, index_records):
        if not embedding:
            continue
        vectorized_ids.append(item.node_id)
        vector_payloads.append({**record, "vector": embedding})
    ingest_err = await _write_dense_sparse_for_document(
        document_id,
        node_count=len(nodes),
        index_records=index_records,
        vector_payloads=vector_payloads,
    )
    if vector_payloads:
        await mark_nodes_vectorized(vectorized_ids)
    await finish_ingest_run(ingest_run_id, error=ingest_err)
    return {
        "success": ingest_err is None,
        "document_id": document_id,
        "ingest_run_id": ingest_run_id,
        "ingestion_mode": "sec_companyfacts",
        "observation_count": len(rows),
        "node_count": len(nodes),
        "vectorized_count": len(vector_payloads),
        "error": ingest_err,
    }


async def process_document(file_path: str, file_type: str, document_id: int) -> dict[str, Any]:
    await ensure_schema()
    title = os.path.basename(file_path)
    path = Path(file_path)
    if _should_ingest_sec_company_facts(file_path, file_type):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {
                "success": False,
                "document_id": document_id,
                "error": f"无法解析 JSON：{exc}",
            }
        if not is_sec_company_facts_payload(data):
            return {
                "success": False,
                "document_id": document_id,
                "error": "JSON 需为 SEC companyfacts 格式（含 cik、facts）",
            }
        await upsert_document(
            document_id,
            title=title,
            source_uri=str(path.resolve()),
            file_type="sec_companyfacts",
            metadata={
                "pipeline_version": config.rag_pipeline_version,
                "domain": "finance",
                "source_kind": "sec_companyfacts",
                "cik": data.get("cik"),
            },
        )
        ingest_run_id = await start_ingest_run(document_id, metadata={"source_uri": str(path.resolve())})
        try:
            return await _ingest_sec_company_facts_json(file_path, document_id, ingest_run_id, data)
        except Exception as exc:
            await finish_ingest_run(ingest_run_id, error=str(exc))
            logger.exception("[Ingestion] SEC companyfacts failed for document {}", document_id)
            return {
                "success": False,
                "document_id": document_id,
                "ingest_run_id": ingest_run_id,
                "error": str(exc),
            }

    await upsert_document(
        document_id,
        title=title,
        source_uri=file_path,
        file_type=file_type,
        metadata={"pipeline_version": config.rag_pipeline_version},
    )
    ingest_run_id = await start_ingest_run(document_id, metadata={"source_uri": file_path})
    try:
        parsed = await parse_document_content(file_path, file_type)
        if not parsed.get("success"):
            raise RuntimeError(parsed.get("error") or "文档解析失败")
        text = (parsed.get("text") or "").strip()
        if not text:
            raise RuntimeError("解析结果为空")

        metadata = dict(parsed.get("metadata") or {})
        out = await _ingest_parsed_text_pipeline(document_id, ingest_run_id, text, metadata)
        out["word_count"] = len(text)
        return out
    except Exception as exc:
        await finish_ingest_run(ingest_run_id, error=str(exc))
        logger.exception("[Ingestion] Failed for document {}", document_id)
        return {
            "success": False,
            "document_id": document_id,
            "ingest_run_id": ingest_run_id,
            "error": str(exc),
        }


async def process_edgar_filing_document(
    file_path: str,
    document_id: int,
    *,
    cik: int,
    accession: str,
    form: str = "",
    filed: str | None = None,
    entity_name: str | None = None,
    source_url: str | None = None,
    primary_document: str | None = None,
) -> dict[str, Any]:
    """Ingest one SEC primary filing HTML; align with companyfacts chunks via finance_accns / finance_forms."""
    await ensure_schema()
    path = Path(file_path)
    title = f"EDGAR {form or '?'} | {accession}"
    uri = source_url or str(path.resolve())
    doc_meta: dict[str, Any] = {
        "pipeline_version": config.rag_pipeline_version,
        "domain": "finance",
        "source_kind": "sec_edgar_filing",
        "cik": cik,
        "sec_accession": accession,
        "sec_filing_date": filed,
        "entity_name": entity_name,
        "primary_document": primary_document,
        # Structured label fields for catalog / UI (see tools.document_display)
        "form": form or None,
    }
    await upsert_document(
        document_id,
        title=title,
        source_uri=uri,
        file_type="sec_edgar_html",
        metadata=doc_meta,
    )
    ingest_run_id = await start_ingest_run(document_id, metadata={"source_uri": uri})
    try:
        body: str
        parse_meta: dict[str, Any] = {}
        if config.edgar_html_use_unstructured:
            try:
                from tools.edgar_unstructured_ingest import (
                    body_text_from_unstructured_json,
                    prepare_edgar_htm_with_unstructured,
                )

                if path.suffix.lower() == ".json" and ".unstructured." in path.name:
                    body = await asyncio.to_thread(body_text_from_unstructured_json, path)
                    if not body.strip():
                        raise RuntimeError("Unstructured JSON 正文为空")
                    parse_meta = {
                        "parser": "unstructured",
                        "unstructured_json": str(path.resolve()),
                        "file_type": "html",
                    }
                else:
                    body, _jp, extra = await asyncio.to_thread(
                        prepare_edgar_htm_with_unstructured,
                        path,
                    )
                    if not body.strip():
                        raise RuntimeError("Unstructured 解析正文为空")
                    parse_meta = {**extra, "file_type": "html"}
            except Exception as unr_exc:
                logger.warning(
                    "[Ingestion] EDGAR Unstructured 不可用或失败，回退内置 HTML 解析: {}",
                    unr_exc,
                )
                parsed = await parse_document_content(str(path), "html")
                if not parsed.get("success"):
                    raise RuntimeError(parsed.get("error") or "HTML 解析失败")
                body = (parsed.get("text") or "").strip()
                if not body:
                    raise RuntimeError("HTML 解析结果为空")
                parse_meta = dict(parsed.get("metadata") or {})
        else:
            parsed = await parse_document_content(str(path), "html")
            if not parsed.get("success"):
                raise RuntimeError(parsed.get("error") or "HTML 解析失败")
            body = (parsed.get("text") or "").strip()
            if not body:
                raise RuntimeError("HTML 解析结果为空")
            parse_meta = dict(parsed.get("metadata") or {})

        uj = parse_meta.get("unstructured_json")
        if uj:
            doc_meta = {**doc_meta, "unstructured_json": uj}
            await upsert_document(
                document_id,
                title=title,
                source_uri=uri,
                file_type="sec_edgar_html",
                metadata=doc_meta,
            )

        header = (
            f"[SEC EDGAR] CIK {cik} | accession {accession} | form {form or 'unknown'}"
            f" | filed {filed or 'n/a'} | doc {primary_document or 'n/a'}\n\n"
        )
        text = header + body
        chunk_meta: dict[str, Any] = {
            "domain": "finance",
            "source_kind": "sec_edgar_filing",
            "cik": cik,
            "entity_name": entity_name,
            "sec_accession": accession,
            "finance_accns": [accession],
        }
        if form:
            chunk_meta["finance_forms"] = [form]
            base_form = _normalize_finance_form(form)
            if base_form:
                chunk_meta["finance_form_base"] = [base_form]
        if filed:
            chunk_meta["sec_filing_date"] = filed
        if uj:
            chunk_meta["unstructured_json"] = uj
        meta = dict(parse_meta)
        meta.update(chunk_meta)
        out = await _ingest_parsed_text_pipeline(document_id, ingest_run_id, text, meta)
        out["ingestion_mode"] = "sec_edgar_html"
        out["word_count"] = len(body)
        if uj:
            out["unstructured_json"] = uj
        return out
    except Exception as exc:
        await finish_ingest_run(ingest_run_id, error=str(exc))
        logger.exception("[Ingestion] EDGAR filing failed for document {}", document_id)
        return {
            "success": False,
            "document_id": document_id,
            "ingest_run_id": ingest_run_id,
            "error": str(exc),
        }


async def reindex_document_vectors(document_id: int) -> dict[str, Any]:
    await ensure_schema()
    nodes = await list_document_nodes(document_id)
    if not nodes:
        return {"success": False, "message": f"Document {document_id} not found"}

    index_records = [_node_index_record(item) for item in nodes]
    embeddings = await generate_embeddings_batch([item["text"] for item in nodes])
    vector_payloads = []
    vectorized_ids: list[str] = []
    for item, embedding, record in zip(nodes, embeddings, index_records):
        if not embedding:
            continue
        vectorized_ids.append(item["node_id"])
        vector_payloads.append({**record, "vector": embedding})
    err = await _write_dense_sparse_for_document(
        document_id,
        node_count=len(nodes),
        index_records=index_records,
        vector_payloads=vector_payloads,
    )
    if vector_payloads:
        await mark_nodes_vectorized(vectorized_ids)
    return {
        "success": err is None,
        "document_id": document_id,
        "total_nodes": len(nodes),
        "vectorized_nodes": len(vector_payloads),
        "error": err,
    }
