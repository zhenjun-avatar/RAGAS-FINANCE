"""Structured sparse query profiles for OpenSearch.

Finance-only deployment. To add another domain later:
- Add keyword fields in ``retrieval_fields.RETRIEVAL_INDEX_KEYWORD_FIELDS``.
- Extend ``build_retrieval_fields`` / ``_extract_domain`` for that domain.
- Add a ``_build_<domain>_plan`` and branch in ``build_sparse_query_plan``.
- Route indices in ``sparse_opensearch._index_for_domain`` / ``_configured_index_names``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from core.config import config

_FINANCE_SECTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("management_discussion", ("md&a", "management discussion", "results of operations", "管理层讨论")),
    ("liquidity", ("liquidity", "capital resources", "流动性", "资本资源")),
    ("risk_factors", ("risk factors", "风险因素")),
    ("business_overview", ("business overview", "业务概览", "业务概况")),
)


@dataclass
class SparseQueryPlan:
    profile: str
    filters: list[dict[str, Any]] = field(default_factory=list)
    must: list[dict[str, Any]] = field(default_factory=list)
    should: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)

    def to_bool_query(self, base_filters: list[dict[str, Any]]) -> dict[str, Any]:
        query: dict[str, Any] = {
            "must": list(self.must),
            "filter": [*base_filters, *self.filters],
        }
        if self.should:
            query["should"] = list(self.should)
        return {"bool": query}


def _match_phrase(field: str, value: str, boost: float) -> dict[str, Any]:
    return {"match_phrase": {field: {"query": value, "boost": boost}}}


def _term_should(field: str, value: Any, boost: float) -> dict[str, Any]:
    return {"term": {field: {"value": value, "boost": boost}}}


def _detect_finance_sections(query: str) -> list[str]:
    lowered = (query or "").lower()
    matched: list[str] = []
    for value, aliases in _FINANCE_SECTION_RULES:
        if any(alias.lower() in lowered for alias in aliases):
            matched.append(value)
    return matched


def _build_finance_plan(
    query: str,
    *,
    narrative_targets: tuple[str, ...] = (),
    term_targets: tuple[str, ...] = (),
) -> SparseQueryPlan:
    normalized = (query or "").strip()
    sections = list(dict.fromkeys([*_detect_finance_sections(normalized), *narrative_targets]))
    should: list[dict[str, Any]] = [
        _term_should("domain", "finance", 7.0),
        _term_should("content_type", "finance_chunk", 2.0),
        _term_should("content_type", "financial_note", 1.4),
        _term_should("content_type", "financial_statement", 1.2),
    ]
    for section in sections:
        should.append(_match_phrase("search_hints", section, 6.0))
    for term in term_targets:
        clean = str(term).strip()
        if not clean:
            continue
        should.append(_match_phrase("search_hints", clean, 4.8))
        should.append(_match_phrase("title", clean, 2.4))
    return SparseQueryPlan(
        profile="finance_v1",
        filters=[{"term": {"domain": "finance"}}],
        must=[
            {
                "multi_match": {
                    "query": normalized,
                    "fields": ["title^2.5", "search_hints^4", "text"],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ],
        should=should,
        slots={
            "domain": "finance",
            "finance_sections": sections,
            "term_targets": [str(term).strip() for term in term_targets if str(term).strip()],
        },
    )


def build_sparse_query_plan(
    query: str,
    *,
    narrative_targets: tuple[str, ...] = (),
    term_targets: tuple[str, ...] = (),
) -> SparseQueryPlan:
    scope = (config.opensearch_sparse_search_scope or "finance").strip().lower()
    if scope == "exam":
        logger.warning(
            "[SparseQuery] OPENSEARCH_SPARSE_SEARCH_SCOPE=exam is removed; using finance profile"
        )
        scope = "finance"

    if scope in ("finance", "all"):
        return _build_finance_plan(
            query,
            narrative_targets=narrative_targets,
            term_targets=term_targets,
        )

    normalized = (query or "").strip()
    return SparseQueryPlan(
        profile="generic_v1",
        must=[
            {
                "multi_match": {
                    "query": normalized,
                    "fields": ["title^2", "search_hints^3", "text"],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ],
    )
