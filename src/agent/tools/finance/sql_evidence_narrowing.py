"""Narrow RAG nodes using SEC SQL observation rows (no re-embedding).

Matching scans node ``text`` plus **flattened metadata**: scalars and nested list/tuple/set
values (e.g. ``finance_accns`` on EDGAR chunks). Company-facts chunks still match via
``accn=...`` / ``taxonomy.metric_key`` in ``text``.

*Strict* (default): output length is ``top_k`` — SQL-matched nodes first (by score), then
backfill with highest-score non-matched until ``top_k``. *Non-strict*: return full pool with
matched nodes ordered before non-matched (caller still slices ``[:top_k]`` for the prompt).
"""

from __future__ import annotations

from typing import Any

# Cap work per node: metadata can contain large lists or deep nesting.
_META_LIST_ITEM_CAP = 512
_META_MAX_DEPTH = 12


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower()


def extract_sql_evidence_signals(
    sql_rows: list[dict[str, Any]],
    *,
    max_accns: int = 24,
    max_tags: int = 40,
) -> tuple[frozenset[str], frozenset[str]]:
    """Unique accession numbers and ``taxonomy.metric_key`` strings from merged SQL rows.

    Rows are consumed in merge order (hint hits first), so caps favor question-relevant facts.
    """
    accns: list[str] = []
    tags: list[str] = []
    seen_a: set[str] = set()
    seen_t: set[str] = set()
    for r in sql_rows:
        if len(seen_a) < max_accns:
            raw = r.get("accn")
            if raw:
                a = str(raw).strip()
                if a and a not in seen_a:
                    seen_a.add(a)
                    accns.append(a)
        if len(seen_t) < max_tags:
            tax = r.get("taxonomy")
            mk = r.get("metric_key")
            if tax and mk:
                tag = f"{tax}.{mk}"
                if tag not in seen_t:
                    seen_t.add(tag)
                    tags.append(tag)
        if len(seen_a) >= max_accns and len(seen_t) >= max_tags:
            break
    return frozenset(accns), frozenset(tags)


def _append_metadata_match_tokens(parts: list[str], value: Any, *, depth: int = 0) -> None:
    """Collect string tokens from metadata for substring matching (lists included; dicts skipped)."""
    if value is None or depth > _META_MAX_DEPTH:
        return
    if isinstance(value, (str, int, float, bool)):
        parts.append(str(value))
        return
    if isinstance(value, (list, tuple, set)):
        for i, item in enumerate(value):
            if i >= _META_LIST_ITEM_CAP:
                break
            _append_metadata_match_tokens(parts, item, depth=depth + 1)


def _collect_string_tokens(out: set[str], value: Any, *, depth: int = 0) -> None:
    if value is None or depth > _META_MAX_DEPTH:
        return
    if isinstance(value, (str, int, float, bool)):
        s = _norm_token(value)
        if s:
            out.add(s)
        return
    if isinstance(value, (list, tuple, set)):
        for i, item in enumerate(value):
            if i >= _META_LIST_ITEM_CAP:
                break
            _collect_string_tokens(out, item, depth=depth + 1)


def _node_structured_sets(node: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return normalized metadata sets for accession and metric-key exact matching."""
    accn_keys = ("finance_accns", "sec_accession")
    metric_keys = ("finance_metric_exact_keys", "finance_metric_keys")
    accns: set[str] = set()
    metrics: set[str] = set()

    for key in accn_keys:
        _collect_string_tokens(accns, node.get(key))
    for key in metric_keys:
        _collect_string_tokens(metrics, node.get(key))

    meta = node.get("metadata")
    if isinstance(meta, dict):
        for key in accn_keys:
            _collect_string_tokens(accns, meta.get(key))
        for key in metric_keys:
            _collect_string_tokens(metrics, meta.get(key))
    return accns, metrics


def _node_text_and_meta_blob(node: dict[str, Any]) -> str:
    parts: list[str] = []
    t = node.get("text")
    if t:
        parts.append(str(t))
    meta = node.get("metadata")
    if isinstance(meta, dict):
        for k, v in meta.items():
            if v is None or str(k).startswith("_"):
                continue
            _append_metadata_match_tokens(parts, v, depth=0)
    return "\n".join(parts)


def _node_match_details(
    node: dict[str, Any],
    accns: frozenset[str],
    tags: frozenset[str],
    metric_keys: frozenset[str],
) -> tuple[bool, bool, bool]:
    """Return (matched, structured_matched, text_matched)."""
    if not accns and not tags:
        return False, False, False

    accns_norm = {_norm_token(a) for a in accns}
    tags_norm = {_norm_token(t) for t in tags}
    metric_norm = {_norm_token(k) for k in metric_keys}

    node_accns, node_metrics = _node_structured_sets(node)
    if accns_norm and node_accns.intersection(accns_norm):
        return True, True, False
    # Structured metric-key matching is the highest-signal route for SQL->RAG alignment.
    if metric_norm and node_metrics.intersection(metric_norm):
        return True, True, False
    if tags_norm and node_metrics.intersection(tags_norm):
        return True, True, False

    blob = _node_text_and_meta_blob(node)
    if not blob.strip():
        return False, False, False
    if any(a in blob for a in accns_norm):
        return True, False, True
    if any(tag in blob for tag in tags_norm):
        return True, False, True
    return False, False, False


def prioritize_nodes_by_sql_evidence(
    nodes: list[dict[str, Any]],
    sql_rows: list[dict[str, Any]],
    *,
    max_accns: int = 24,
    max_tags: int = 40,
    strict: bool = True,
    top_k: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """SQL-matched nodes first; strict mode caps list to ``top_k`` with score-based backfill."""
    if not nodes or not sql_rows:
        return list(nodes), {"applied": False, "reason": "empty_nodes_or_sql_rows"}

    accns, tags = extract_sql_evidence_signals(sql_rows, max_accns=max_accns, max_tags=max_tags)
    if not accns and not tags:
        return list(nodes), {"applied": False, "reason": "no_signals"}
    metric_keys = frozenset(tag.split(".")[-1] for tag in tags if "." in tag)

    match_details = [_node_match_details(n, accns, tags, metric_keys) for n in nodes]
    hits = [m[0] for m in match_details]
    structured_hit_count = sum(1 for _, structured, _ in match_details if structured)
    text_hit_count = sum(1 for _, _, text in match_details if text)
    matched_nodes = [n for n, h in zip(nodes, hits, strict=True) if h]
    unmatched_nodes = [n for n, h in zip(nodes, hits, strict=True) if not h]
    score_key = lambda n: -float(n.get("score") or 0.0)
    matched_nodes.sort(key=score_key)
    unmatched_nodes.sort(key=score_key)

    k = max(1, int(top_k))
    matched_used: int | None = None
    backfill_used: int | None = None
    if strict:
        if len(matched_nodes) >= k:
            ordered = matched_nodes[:k]
            backfill_used = 0
            matched_used = k
        else:
            need = k - len(matched_nodes)
            ordered = matched_nodes + unmatched_nodes[:need]
            backfill_used = min(need, len(unmatched_nodes))
            matched_used = len(matched_nodes)
    else:
        ordered = matched_nodes + unmatched_nodes

    stats: dict[str, Any] = {
        "applied": True,
        "strict": strict,
        "top_k": k,
        "signal_accn_count": len(accns),
        "signal_tag_count": len(tags),
        "nodes_in": len(nodes),
        # Backward-compat key: keep old meaning as total matched count.
        "nodes_text_matched_in_pool": len(matched_nodes),
        "nodes_structured_matched_in_pool": structured_hit_count,
        "nodes_text_fallback_matched_in_pool": text_hit_count,
        "pool_out": len(ordered),
        "strict_matched_used": matched_used,
        "strict_backfill_used": backfill_used,
        "sample_accns": sorted(accns)[:5],
        "sample_tags": sorted(tags)[:8],
    }
    return ordered, stats
