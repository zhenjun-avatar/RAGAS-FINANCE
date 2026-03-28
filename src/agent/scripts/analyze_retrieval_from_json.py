#!/usr/bin/env python3
"""Analyze retrieval JSON and render ranking/score comparison artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_json_any_encoding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    raise ValueError(f"Cannot decode JSON file: {path}")


def _ranked(items: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(items, key=lambda x: float(x.get(score_key) or 0.0), reverse=True)


def _get_score(item: dict[str, Any]) -> float:
    for k in ("rerank_score", "fusion_score", "dense_score", "sparse_score"):
        v = item.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return 0.0


def _to_map(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for x in items:
        nid = x.get("node_id")
        if nid:
            out[str(nid)] = x
    return out


def _md_table(title: str, rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    parts = [f"## {title}", "", "| " + " | ".join(c[1] for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        line = []
        for key, _ in cols:
            val = r.get(key, "")
            if isinstance(val, float):
                line.append(f"{val:.6f}")
            else:
                line.append(str(val))
        parts.append("| " + " | ".join(line) + " |")
    parts.append("")
    return "\n".join(parts)


async def _recompute_if_needed(
    data: dict[str, Any],
    pipeline_trace: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rerank_hits = pipeline_trace.get("retrieval_rerank_hits") or {}
    inp = rerank_hits.get("input") or []
    out = rerank_hits.get("output") or []
    if inp and out:
        return inp, out, "from_pipeline_trace"

    # fallback: recompute retrieval debug from question + inferred document_ids
    question = data.get("question") or ""
    doc_ids: set[int] = set()
    for section in ("retrieval_dense_hits", "retrieval_sparse_hits"):
        sec = pipeline_trace.get(section) or {}
        for stage in ("summary", "leaf"):
            for item in sec.get(stage) or []:
                did = item.get("document_id")
                if did is not None:
                    try:
                        doc_ids.add(int(did))
                    except Exception:
                        pass
    if not question or not doc_ids:
        return [], [], "missing_inputs"

    from tools.llamaindex_retrieval import retrieval_service  # local import to avoid startup overhead

    result = await retrieval_service.retrieve(query=question, document_ids=sorted(doc_ids))
    dbg = result.get("debug") or {}
    pre = dbg.get("pre_rerank") or []
    rer = dbg.get("reranked") or []
    return pre, rer, "recomputed_from_retriever"


def _plot_rank_change(pre: list[dict[str, Any]], post: list[dict[str, Any]], out_png: Path) -> None:
    pre_rank = {x["node_id"]: i + 1 for i, x in enumerate(pre) if x.get("node_id")}
    post_rank = {x["node_id"]: i + 1 for i, x in enumerate(post) if x.get("node_id")}
    common = [nid for nid in pre_rank if nid in post_rank]
    if not common:
        return

    # only top 20 by best stage rank
    common = sorted(common, key=lambda nid: min(pre_rank[nid], post_rank[nid]))[:20]
    x0, x1 = 0, 1

    fig, ax = plt.subplots(figsize=(10, 8))
    for nid in common:
        y0, y1 = pre_rank[nid], post_rank[nid]
        color = "#2E7D32" if y1 < y0 else ("#C62828" if y1 > y0 else "#616161")
        ax.plot([x0, x1], [y0, y1], color=color, alpha=0.8)
        ax.text(x0 - 0.02, y0, str(y0), ha="right", va="center", fontsize=8)
        ax.text(x1 + 0.02, y1, str(y1), ha="left", va="center", fontsize=8)

    ax.set_xticks([x0, x1], ["pre_rerank", "rerank_output"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(max(max(pre_rank.values()), max(post_rank.values())) + 1, 0)
    ax.set_ylabel("rank (smaller is better)")
    ax.set_title("Node rank change: pre_rerank -> rerank_output")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)


def _collect_node_text_map(*groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rows in groups:
        for r in rows:
            nid = r.get("node_id")
            if not nid:
                continue
            node_id = str(nid)
            if node_id not in out:
                out[node_id] = {
                    "title": r.get("title") or "",
                    "text_preview": r.get("text_preview") or r.get("text") or "",
                }
            else:
                if not out[node_id].get("text_preview"):
                    out[node_id]["text_preview"] = r.get("text_preview") or r.get("text") or ""
    return out


def _write_interactive_html(
    out_html: Path,
    *,
    question: str,
    source_json: Path,
    dense_summary: list[dict[str, Any]],
    sparse_summary: list[dict[str, Any]],
    dense_leaf: list[dict[str, Any]],
    sparse_leaf: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    node_text_map: dict[str, dict[str, Any]],
) -> None:
    dense_score_by_id = {str(x.get("node_id")): x.get("dense_score") for x in (dense_summary + dense_leaf) if x.get("node_id")}
    sparse_score_by_id = {str(x.get("node_id")): x.get("sparse_score") for x in (sparse_summary + sparse_leaf) if x.get("node_id")}
    rows = []
    for i, row in enumerate(delta_rows, start=1):
        nid = str(row.get("node_id"))
        rows.append(
            {
                "idx": i,
                "node_id": nid,
                "title": row.get("title") or "",
                "level": row.get("level"),
                "pre_rank": row.get("pre_rank"),
                "post_rank": row.get("post_rank"),
                "dense_score": dense_score_by_id.get(nid),
                "sparse_score": sparse_score_by_id.get(nid),
                "rerank_pre_score": row.get("pre_score"),
                "rerank_post_score": row.get("post_score"),
                "rank_change": row.get("rank_change(pre-post)"),
            }
        )

    payload = {
        "rows": rows,
        "node_text_map": node_text_map,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Retrieval Interactive Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .grid {{ display: grid; grid-template-columns: 58% 42%; gap: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; }}
    th {{ background: #f6f6f6; position: sticky; top: 0; }}
    tr:hover {{ background: #f9f9ff; }}
    .clickable {{ color: #0b61d8; cursor: pointer; text-decoration: underline; }}
    .panel {{ border: 1px solid #ddd; padding: 10px; border-radius: 6px; }}
    .mono {{ font-family: Consolas, monospace; font-size: 12px; }}
    .content {{ white-space: pre-wrap; line-height: 1.4; max-height: 72vh; overflow: auto; }}
  </style>
</head>
<body>
  <h2>Retrieval 交互对比报告</h2>
  <div>source: <span class="mono">{source_json}</span></div>
  <div>question: {question}</div>
  <p>点击左侧序号可查看节点内容；表格中已标注 dense/sparse/rerank 的 score。</p>
  <div class="grid">
    <div style="max-height: 80vh; overflow: auto;">
      <table id="tbl">
        <thead>
          <tr>
            <th>#</th><th>node_id</th><th>title</th><th>level</th>
            <th>dense_score</th><th>sparse_score</th>
            <th>pre_rank</th><th>pre_score</th>
            <th>post_rank</th><th>post_score</th><th>rank_change</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="panel">
      <div><b>选中节点</b></div>
      <div id="meta" class="mono">点击左侧序号查看详情</div>
      <hr />
      <div id="content" class="content"></div>
    </div>
  </div>
  <script>
    const data = {payload_json};
    const tbody = document.querySelector('#tbl tbody');
    const meta = document.getElementById('meta');
    const content = document.getElementById('content');

    function fmt(v) {{
      if (v === null || v === undefined || v === '') return '';
      if (typeof v === 'number') return Number(v).toFixed(6);
      return String(v);
    }}

    function showRow(r) {{
      const t = data.node_text_map[r.node_id] || {{}};
      const text = (t.text_preview || '').trim();
      meta.textContent = `node_id=${{r.node_id}} | level=${{r.level}} | dense=${{fmt(r.dense_score)}} | sparse=${{fmt(r.sparse_score)}} | pre=${{fmt(r.rerank_pre_score)}} | post=${{fmt(r.rerank_post_score)}}`;
      content.textContent = text || '(无内容)';
    }}

    for (const r of data.rows) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="clickable">${{r.idx}}</span></td>
        <td class="mono">${{r.node_id}}</td>
        <td>${{r.title || ''}}</td>
        <td>${{r.level ?? ''}}</td>
        <td>${{fmt(r.dense_score)}}</td>
        <td>${{fmt(r.sparse_score)}}</td>
        <td>${{r.pre_rank}}</td>
        <td>${{fmt(r.rerank_pre_score)}}</td>
        <td>${{r.post_rank}}</td>
        <td>${{fmt(r.rerank_post_score)}}</td>
        <td>${{r.rank_change}}</td>
      `;
      tr.querySelector('.clickable').addEventListener('click', () => showRow(r));
      tbody.appendChild(tr);
    }}
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval comparison from ask JSON")
    parser.add_argument("--input", required=True, help="ask json path")
    parser.add_argument("--out-prefix", default="", help="output prefix path (without extension)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    data = _load_json_any_encoding(input_path)
    pt = data.get("pipeline_trace") or {}
    out_prefix = Path(args.out_prefix).resolve() if args.out_prefix else input_path.with_name(input_path.stem + "_analysis")

    dense = pt.get("retrieval_dense_hits") or {}
    sparse = pt.get("retrieval_sparse_hits") or {}
    dense_summary = _ranked(dense.get("summary") or [], "dense_score")
    dense_leaf = _ranked(dense.get("leaf") or [], "dense_score")
    sparse_summary = _ranked(sparse.get("summary") or [], "sparse_score")
    sparse_leaf = _ranked(sparse.get("leaf") or [], "sparse_score")

    pre_rerank, rerank_out, rerank_source = await _recompute_if_needed(data, pt)
    pre_rerank = sorted(pre_rerank, key=_get_score, reverse=True)
    rerank_out = sorted(rerank_out, key=_get_score, reverse=True)

    # rank delta table
    pre_rank = {x["node_id"]: i + 1 for i, x in enumerate(pre_rerank) if x.get("node_id")}
    post_rank = {x["node_id"]: i + 1 for i, x in enumerate(rerank_out) if x.get("node_id")}
    node_ids = sorted(set(pre_rank) | set(post_rank), key=lambda nid: (post_rank.get(nid, 10**9), pre_rank.get(nid, 10**9)))
    pre_map = _to_map(pre_rerank)
    post_map = _to_map(rerank_out)
    delta_rows: list[dict[str, Any]] = []
    for nid in node_ids:
        p = pre_rank.get(nid)
        q = post_rank.get(nid)
        base = post_map.get(nid) or pre_map.get(nid) or {}
        delta = None if (p is None or q is None) else (p - q)
        delta_rows.append(
            {
                "node_id": nid,
                "document_id": base.get("document_id"),
                "level": base.get("level"),
                "title": base.get("title"),
                "pre_rank": p if p is not None else "-",
                "pre_score": _get_score(pre_map.get(nid) or {}) if p is not None else 0.0,
                "post_rank": q if q is not None else "-",
                "post_score": _get_score(post_map.get(nid) or {}) if q is not None else 0.0,
                "rank_change(pre-post)": delta if delta is not None else "dropped_or_new",
            }
        )

    # Write markdown summary
    md_parts = [
        f"# Retrieval Node Comparison",
        "",
        f"- source_json: `{input_path}`",
        f"- rerank_source: `{rerank_source}`",
        f"- question: {data.get('question','')}",
        "",
    ]

    cols_dense = [("node_id", "node_id"), ("document_id", "doc_id"), ("level", "level"), ("dense_score", "dense_score"), ("title", "title")]
    cols_sparse = [("node_id", "node_id"), ("document_id", "doc_id"), ("level", "level"), ("sparse_score", "sparse_score"), ("title", "title")]
    cols_delta = [
        ("node_id", "node_id"),
        ("document_id", "doc_id"),
        ("level", "level"),
        ("title", "title"),
        ("pre_rank", "pre_rank"),
        ("pre_score", "pre_score"),
        ("post_rank", "post_rank"),
        ("post_score", "post_score"),
        ("rank_change(pre-post)", "rank_change(pre-post)"),
    ]
    md_parts.append(_md_table("Dense Summary", dense_summary, cols_dense))
    md_parts.append(_md_table("Sparse Summary", sparse_summary, cols_sparse))
    md_parts.append(_md_table("Dense Leaf", dense_leaf, cols_dense))
    md_parts.append(_md_table("Sparse Leaf", sparse_leaf, cols_sparse))
    md_parts.append(_md_table("Rerank Input vs Output", delta_rows, cols_delta))

    out_md = out_prefix.with_suffix(".md")
    out_md.write_text("\n".join(md_parts), encoding="utf-8")

    # Write machine-readable JSON
    out_json = out_prefix.with_suffix(".json")
    out_json.write_text(
        json.dumps(
            {
                "source_json": str(input_path),
                "rerank_source": rerank_source,
                "dense_summary": dense_summary,
                "sparse_summary": sparse_summary,
                "dense_leaf": dense_leaf,
                "sparse_leaf": sparse_leaf,
                "rerank_compare": delta_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_png = out_prefix.with_suffix(".png")
    _plot_rank_change(pre_rerank, rerank_out, out_png)

    node_text_map = _collect_node_text_map(
        dense_summary,
        sparse_summary,
        dense_leaf,
        sparse_leaf,
        pre_rerank,
        rerank_out,
    )
    out_html = out_prefix.with_suffix(".html")
    _write_interactive_html(
        out_html,
        question=data.get("question", ""),
        source_json=input_path,
        dense_summary=dense_summary,
        sparse_summary=sparse_summary,
        dense_leaf=dense_leaf,
        sparse_leaf=sparse_leaf,
        delta_rows=delta_rows,
        node_text_map=node_text_map,
    )

    print(f"saved_md={out_md}")
    print(f"saved_json={out_json}")
    print(f"saved_png={out_png}")
    print(f"saved_html={out_html}")


if __name__ == "__main__":
    asyncio.run(main())
