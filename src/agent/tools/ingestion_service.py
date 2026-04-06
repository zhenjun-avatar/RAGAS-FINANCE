"""Document ingestion pipeline for node-centric RAG storage."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from collections import defaultdict

from loguru import logger

from core.config import config
from .chunk_segmenter import ChunkPayload, build_retrieval_chunks
from .document_parser import parse_document_content
from .finance.financial_facts_repository import replace_sec_observations
from .finance.companyfacts_accession_period import merge_companyfacts_period_into_metadata
from .finance.sec_company_facts import (
    batch_lines_for_nodes,
    batch_row_groups_for_nodes,
    build_chunk_filter_metadata,
    flatten_sec_company_facts,
    is_sec_company_facts_payload,
)
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
    "form",
    "document_fiscal_year",
    "document_fiscal_period",
    # retrieval filters
    "finance_forms",
    "finance_form_base",
    "finance_accns",
    "finance_metric_exact_keys",
    "finance_period",
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


# Maximum chars stored in a section node's text field.
# Prevents index bloat for very large sections while keeping content searchable.
_SECTION_TEXT_CAP = 6000


def _merge_chunk_metadata(document_metadata: dict[str, Any], chunk: ChunkPayload) -> dict[str, Any]:
    merged = _base_node_metadata(document_metadata)
    merged.update(chunk.metadata)
    return merged


def _chunk_section_path(chunk: ChunkPayload) -> list[str]:
    raw = chunk.metadata.get("section_path")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()][:12]
    raw = chunk.metadata.get("heading_path")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()][:12]
    section_title = str(chunk.metadata.get("section_title") or "").strip()
    return [section_title] if section_title else []


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


def _build_section_tree_nodes(
    chunks: list[ChunkPayload],
    *,
    document_id: int,
    ingest_run_id: str,
    metadata: dict[str, Any],
) -> list[NodeRecord]:
    """Build a multi-level section tree from chunk section_paths.

    Node levels:
      level = 0          : leaf chunk (verbatim text; primary retrieval and context unit)
      level = path_depth : section node, where depth = len(section_path)
                           level=1 is shallowest (e.g. "Part I"),
                           higher levels are progressively deeper and more specific
                           (e.g. level=3 for "Liquidity and Capital Resources").

    Section node text = concatenation of all descendant leaf texts, capped at
    _SECTION_TEXT_CAP chars.  No LLM calls are needed.

    This structure enables:
      - Inner-to-outer retrieval: search at deeper levels first, fall back to shallower.
      - Section-bounded context: fetch siblings (same parent_id) instead of ±radius neighbors.
      - Precise section_role filtering once section nodes are indexed.
    """
    # 1. Collect section path per chunk
    chunk_paths: list[tuple[str, ...]] = [
        tuple(_chunk_section_path(c)) for c in chunks
    ]

    # 2. Build ordered map of all unique path prefixes (document order = first-seen)
    prefix_order: dict[tuple[str, ...], int] = {}
    for path in chunk_paths:
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            if prefix not in prefix_order:
                prefix_order[prefix] = len(prefix_order)

    # 3. Group leaf texts by full section_path for section node text assembly
    leaves_by_path: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for chunk, path in zip(chunks, chunk_paths):
        if path:
            leaves_by_path[path].append(chunk.text)

    def _subtree_text(prefix: tuple[str, ...]) -> str:
        """Concatenated text of all leaves whose path starts with this prefix."""
        parts: list[str] = []
        for path, texts in leaves_by_path.items():
            if path[: len(prefix)] == prefix:
                parts.extend(texts)
        return "\n\n".join(parts)[:_SECTION_TEXT_CAP]

    # 4. Create section nodes in document order; parent always created before children
    section_id_map: dict[tuple[str, ...], str] = {}
    nodes: list[NodeRecord] = []
    for prefix, order_idx in sorted(prefix_order.items(), key=lambda kv: kv[1]):
        node_id = str(uuid.uuid4())
        section_id_map[prefix] = node_id
        depth = len(prefix)
        title = prefix[-1]
        text = _subtree_text(prefix)
        sec_meta = _base_node_metadata(metadata)
        sec_meta["section_path"] = list(prefix)
        sec_meta["section_depth"] = depth
        nodes.append(
            NodeRecord(
                node_id=node_id,
                document_id=document_id,
                ingest_run_id=ingest_run_id,
                parent_id=section_id_map.get(prefix[:-1]),
                node_type="section",
                level=depth,
                order_index=order_idx,
                title=title,
                text=text or title,
                metadata=_attach_retrieval_fields(
                    node_type="section",
                    level=depth,
                    title=title,
                    text=text or title,
                    metadata=sec_meta,
                ),
            )
        )

    # 5. Leaf chunk nodes (level=0); order continues after all section nodes
    leaf_offset = len(prefix_order)
    for idx, (chunk, path) in enumerate(zip(chunks, chunk_paths)):
        parent_id = section_id_map.get(path)
        title = chunk.title or (path[-1] if path else f"Chunk {idx + 1}")
        chunk_meta = _merge_chunk_metadata(metadata, chunk)
        nodes.append(
            NodeRecord(
                node_id=str(uuid.uuid4()),
                document_id=document_id,
                ingest_run_id=ingest_run_id,
                parent_id=parent_id,
                node_type="chunk",
                level=0,
                order_index=leaf_offset + idx,
                title=title,
                text=chunk.text,
                metadata=_attach_retrieval_fields(
                    node_type="chunk",
                    level=0,
                    title=title,
                    text=chunk.text,
                    metadata=chunk_meta,
                ),
            )
        )

    return nodes


def _build_nodes(document_id: int, ingest_run_id: str, text: str, metadata: dict[str, Any]) -> list[NodeRecord]:
    chunks = build_retrieval_chunks(text)
    return _build_section_tree_nodes(chunks, document_id=document_id, ingest_run_id=ingest_run_id, metadata=metadata)


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


def _elements_to_chunks(
    elements: list[Any],
    base_meta: dict[str, Any],
) -> list[ChunkPayload]:
    """Convert ParsedElement list → ChunkPayload list (1:1, preserving parser boundaries).

    Each element already has clean text and section metadata from the BS4 enricher;
    no further splitting is needed.  Element-level metadata (finance_section, footnotes,
    table_headers_from_narrative …) overrides *base_meta* on a per-key basis.
    """
    chunks: list[ChunkPayload] = []
    for el in elements:
        meta = dict(base_meta)
        # Element-level keys win (e.g. finance_section, finance_accns, content_type)
        meta.update({k: v for k, v in el.metadata.items() if v is not None})
        meta["section_path"] = el.section_path
        title = el.table_name or (el.section_path[-1] if el.section_path else None)
        if title:
            meta["section_title"] = title
        chunks.append(ChunkPayload(text=el.text_for_retrieval, title=title, metadata=meta))
    return chunks


async def _store_nodes(
    document_id: int,
    ingest_run_id: str,
    nodes: list[NodeRecord],
) -> tuple[int, str | None]:
    """Persist nodes to Postgres + Qdrant + OpenSearch. Returns (vectorized_count, error)."""
    await replace_document_nodes(document_id, ingest_run_id, nodes)
    index_records = [_node_index_record(item) for item in nodes]
    embeddings = await generate_embeddings_batch([item.text for item in nodes])
    vector_payloads: list[dict[str, Any]] = []
    vectorized_ids: list[str] = []
    for item, embedding, record in zip(nodes, embeddings, index_records):
        if not embedding:
            continue
        vectorized_ids.append(item.node_id)
        vector_payloads.append({**record, "vector": embedding})
    err = await _write_dense_sparse_for_document(
        document_id,
        node_count=len(nodes),
        index_records=index_records,
        vector_payloads=vector_payloads,
    )
    if vector_payloads:
        await mark_nodes_vectorized(vectorized_ids)
    return len(vector_payloads), err


async def _ingest_parsed_text_pipeline(
    document_id: int,
    ingest_run_id: str,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    nodes = _build_nodes(document_id, ingest_run_id, text, metadata)
    vectorized, ingest_err = await _store_nodes(document_id, ingest_run_id, nodes)
    await finish_ingest_run(ingest_run_id, error=ingest_err)
    return {
        "success": ingest_err is None,
        "document_id": document_id,
        "ingest_run_id": ingest_run_id,
        "node_count": len(nodes),
        "vectorized_count": vectorized,
        "word_count": len(text),
        "metadata": metadata,
        "error": ingest_err,
    }


async def _ingest_chunks_pipeline(
    document_id: int,
    ingest_run_id: str,
    chunks: list[ChunkPayload],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Ingest pre-built chunks (e.g. from edgar_htm_parser) without re-splitting."""
    nodes = _build_section_tree_nodes(chunks, document_id=document_id, ingest_run_id=ingest_run_id, metadata=metadata)
    vectorized, ingest_err = await _store_nodes(document_id, ingest_run_id, nodes)
    await finish_ingest_run(ingest_run_id, error=ingest_err)
    return {
        "success": ingest_err is None,
        "document_id": document_id,
        "ingest_run_id": ingest_run_id,
        "node_count": len(nodes),
        "vectorized_count": vectorized,
        "word_count": sum(len(c.text) for c in chunks),
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
    doc_meta = merge_companyfacts_period_into_metadata(doc_meta, cik=cik, accession=accession)
    if doc_meta.get("form"):
        nb = _normalize_finance_form(str(doc_meta["form"]))
        if nb:
            doc_meta["form"] = nb
    await upsert_document(
        document_id,
        title=title,
        source_uri=uri,
        file_type="sec_edgar_html",
        metadata=doc_meta,
    )
    ingest_run_id = await start_ingest_run(document_id, metadata={"source_uri": uri})
    try:
        # ── BS4 element-aware parser (primary path for .htm/.html) ──────────────
        if path.suffix.lower() in (".htm", ".html") and config.edgar_html_use_bs4_parser:
            from .edgar_htm_enricher import EdgarEnricher
            from .edgar_htm_to_final import parse_htm_to_elements

            form_type = (doc_meta.get("form") or form or "10-q").lower().replace("/a", "")
            elements = await asyncio.to_thread(
                parse_htm_to_elements, path, accession=accession, form_type=form_type,
            )
            elements = await asyncio.to_thread(EdgarEnricher(use_llm=False).enrich, elements)

            chunk_meta: dict[str, Any] = {
                "domain": "finance",
                "source_kind": "sec_edgar_filing",
                "cik": cik,
                "entity_name": entity_name,
                "sec_accession": accession,
                "finance_accns": [accession],
                "language": "en",
                "parser": "edgar_htm_bs4",
            }
            eff_form = doc_meta.get("form") or form
            if eff_form:
                chunk_meta["finance_forms"] = [eff_form]
                if nb := _normalize_finance_form(str(eff_form)):
                    chunk_meta["finance_form_base"] = [nb]
            if filed:
                chunk_meta["sec_filing_date"] = filed

            meta = {**doc_meta, **chunk_meta}
            chunks = _elements_to_chunks(elements, meta)
            out = await _ingest_chunks_pipeline(document_id, ingest_run_id, chunks, meta)
            out["ingestion_mode"] = "sec_edgar_htm_bs4"
            out["element_count"] = len(elements)
            return out

        # ── Legacy paths: unstructured / generic HTML ───────────────────────────
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
        eff_form = doc_meta.get("form") or form
        if eff_form:
            chunk_meta["finance_forms"] = [eff_form]
            base_form = _normalize_finance_form(str(eff_form))
            if base_form:
                chunk_meta["finance_form_base"] = [base_form]
        if filed:
            chunk_meta["sec_filing_date"] = filed
        if uj:
            chunk_meta["unstructured_json"] = uj
        meta = dict(doc_meta)
        meta.update(parse_meta)
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
