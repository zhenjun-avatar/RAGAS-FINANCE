"""Multi-query Bocha rerank for narrative retrieval: merge scores across facet queries.

Keeps llamaindex_retrieval thin; all sub-query wording and merge policy live here.
"""

from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    async def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
        out_stats: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = (q or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def narrative_rerank_subqueries(
    base_rerank_query: str,
    plan_debug: dict[str, Any],
    *,
    enabled: bool,
    max_queries: int,
) -> list[str]:
    """Return 1..max_queries strings for parallel rerank facets."""
    base = (base_rerank_query or "").strip()
    if not base or not enabled:
        return [base] if base else [""]

    cap = max(1, min(int(max_queries), 6))
    raw_targets = plan_debug.get("narrative_targets") or ()
    nt = {str(x).strip() for x in raw_targets if str(x).strip()}

    queries: list[str] = []

    if "risk_factors" in nt:
        queries = [
            base,
            f"{base}\nFocus: distribution through carriers, wholesalers, retailers, resellers, channel dependence.",
            f"{base}\nFocus: trade receivables, counterparties, credit risk, supplier or customer payment exposure.",
        ]
    elif nt & {"margin_cost_structure", "management_discussion"}:
        queries = [
            base,
            f"{base}\nFocus: gross margin, operating expenses, cost of revenue and results of operations drivers.",
        ]
    else:
        return [base]

    out = _dedupe_queries(queries)
    return out[:cap] if out else [base]


async def run_multi_query_rerank(
    reranker: Reranker,
    *,
    queries: list[str],
    candidates: list[dict[str, Any]],
    rerank_keep: int,
    out_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rerank the same pool with each query; keep each node's max rerank_score, then top rerank_keep."""
    keep = max(1, min(int(rerank_keep), len(candidates)))
    qlist = [q for q in queries if (q or "").strip()] or [""]

    if len(qlist) == 1:
        return await reranker.rerank(
            query=qlist[0],
            candidates=candidates,
            top_n=keep,
            out_stats=out_stats,
        )

    per = max(keep // len(qlist), min(12, keep))
    per = min(per, len(candidates))

    best_row: dict[str, dict[str, Any]] = {}
    best_sc: dict[str, float] = {}
    sub_runs: list[dict[str, Any]] = []

    for i, q in enumerate(qlist):
        branch: dict[str, Any] = {}
        ranked = await reranker.rerank(
            query=q.strip(),
            candidates=candidates,
            top_n=per,
            out_stats=branch,
        )
        sub_runs.append(
            {
                "index": i,
                "query_preview": q.strip()[:240],
                "mode": branch.get("mode"),
                "candidates_out": branch.get("candidates_out"),
                "remote_http_called": branch.get("remote_http_called"),
            }
        )
        for item in ranked:
            nid = str(item.get("node_id") or "").strip()
            if not nid:
                continue
            sc = float(item.get("rerank_score") or 0.0)
            if sc > best_sc.get(nid, -1.0):
                best_sc[nid] = sc
                best_row[nid] = dict(item)

    merged = sorted(best_row.values(), key=lambda x: -float(x.get("rerank_score") or 0.0))
    out = merged[:keep]

    out_stats.clear()
    out_stats.update(
        {
            "step": "bocha_rerank_multi",
            "subquery_count": len(qlist),
            "per_sub_top_n": per,
            "candidates_in": len(candidates),
            "candidates_out": len(out),
            "merged_unique_before_trim": len(best_row),
            "sub_runs": sub_runs,
            "mode": sub_runs[-1].get("mode") if sub_runs else None,
            "remote_http_called": any(s.get("remote_http_called") for s in sub_runs),
        }
    )

    return out if out else candidates[:keep]
