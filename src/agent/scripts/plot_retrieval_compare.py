#!/usr/bin/env python3
"""Plot dense/sparse retrieval comparison from ask pipeline_trace JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    raise ValueError(f"Cannot decode JSON file: {path}")


def _get(d: dict[str, Any], key: str, default: int = 0) -> int:
    value = d.get(key, default)
    try:
        return int(value)
    except Exception:
        return default


def _flow_note(pt: dict[str, Any], counts: dict[str, Any]) -> str:
    rr = pt.get("retrieval_rerank_compare") or {}
    if rr:
        add_n = len(rr.get("added_from_summary_ids") or [])
        drop_n = len(rr.get("dropped_by_rerank_ids") or [])
        return f"pre_rerank = leaf_fused + summary_children; added_from_summary={add_n}, rerank_dropped={drop_n}"

    # Backward compatible fallback when old JSON has no retrieval_rerank_compare
    pre = _get(counts, "pre_rerank_pool")
    leaf_union = _get((pt.get("retrieval_overlap") or {}).get("leaf") or {}, "union_count")
    delta = pre - leaf_union
    return (
        "retrieval_rerank_compare not found in this JSON. "
        f"Estimated summary_children contribution ~= pre_rerank_pool({pre}) - leaf_union({leaf_union}) = {delta}."
    )


def plot(trace_json: Path, out_png: Path, out_txt: Path) -> None:
    data = _load_json(trace_json)
    pt = data.get("pipeline_trace") or {}
    counts = pt.get("retrieval_counts") or {}
    overlap = pt.get("retrieval_overlap") or {}

    s = overlap.get("summary") or {}
    l = overlap.get("leaf") or {}

    s_inter = _get(s, "intersection_count")
    s_union = _get(s, "union_count")
    s_dense_only = len(s.get("dense_only_ids") or [])
    s_sparse_only = len(s.get("sparse_only_ids") or [])

    l_inter = _get(l, "intersection_count")
    l_union = _get(l, "union_count")
    l_dense_only = len(l.get("dense_only_ids") or [])
    l_sparse_only = len(l.get("sparse_only_ids") or [])

    summary_dense = _get(counts, "summary_dense")
    summary_sparse = _get(counts, "summary_sparse")
    leaf_dense = _get(counts, "leaf_dense")
    leaf_sparse = _get(counts, "leaf_sparse")
    pre_rerank = _get(counts, "pre_rerank_pool")
    final_ranked = _get(counts, "final_ranked")
    expanded = _get(counts, "context_nodes_expanded")

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.0, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Top-left: overlap stacked bars
    labels = ["summary", "leaf"]
    inter_vals = [s_inter, l_inter]
    dense_only_vals = [s_dense_only, l_dense_only]
    sparse_only_vals = [s_sparse_only, l_sparse_only]

    x = [0, 1]
    ax1.bar(x, inter_vals, color="#4CAF50", label="intersection")
    ax1.bar(x, dense_only_vals, bottom=inter_vals, color="#42A5F5", label="dense_only")
    ax1.bar(
        x,
        sparse_only_vals,
        bottom=[inter_vals[i] + dense_only_vals[i] for i in range(2)],
        color="#FF9800",
        label="sparse_only",
    )
    ax1.set_xticks(x, labels)
    ax1.set_title("Dense vs Sparse overlap")
    ax1.set_ylabel("node count")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.2)

    ax1.text(
        0,
        inter_vals[0] + dense_only_vals[0] + sparse_only_vals[0] + 0.2,
        f"J={float((s.get('jaccard') or 0)):.2f}\nD={summary_dense} S={summary_sparse} U={s_union}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax1.text(
        1,
        inter_vals[1] + dense_only_vals[1] + sparse_only_vals[1] + 0.2,
        f"J={float((l.get('jaccard') or 0)):.2f}\nD={leaf_dense} S={leaf_sparse} U={l_union}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # Top-right: per-source raw counts
    ax2.bar(["summary_dense", "summary_sparse", "leaf_dense", "leaf_sparse"], [summary_dense, summary_sparse, leaf_dense, leaf_sparse], color=["#1565C0", "#EF6C00", "#1565C0", "#EF6C00"])
    ax2.set_title("Raw retrieval counts")
    ax2.set_ylabel("count")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.2)

    # Bottom: pipeline flow
    flow_labels = ["leaf_union", "pre_rerank_pool", "final_ranked", "context_expanded"]
    flow_vals = [l_union, pre_rerank, final_ranked, expanded]
    ax3.plot(flow_labels, flow_vals, marker="o", linewidth=2, color="#7E57C2")
    for i, v in enumerate(flow_vals):
        ax3.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    ax3.set_title("Candidate flow to rerank and final context")
    ax3.set_ylabel("count")
    ax3.grid(axis="y", alpha=0.2)

    note = _flow_note(pt, counts)
    ax3.text(0.01, 0.03, note, transform=ax3.transAxes, fontsize=9, va="bottom")

    q = data.get("question") or ""
    trace_id = data.get("trace_id") or "-"
    fig.suptitle(f"RAG retrieval comparison\nquestion={q} | trace_id={trace_id}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160)

    report = {
        "question": q,
        "trace_id": trace_id,
        "summary": {
            "dense": summary_dense,
            "sparse": summary_sparse,
            "intersection": s_inter,
            "union": s_union,
            "jaccard": float(s.get("jaccard") or 0.0),
        },
        "leaf": {
            "dense": leaf_dense,
            "sparse": leaf_sparse,
            "intersection": l_inter,
            "union": l_union,
            "jaccard": float(l.get("jaccard") or 0.0),
        },
        "flow": {
            "leaf_union": l_union,
            "pre_rerank_pool": pre_rerank,
            "final_ranked": final_ranked,
            "context_nodes_expanded": expanded,
            "note": note,
        },
    }
    out_txt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dense/sparse retrieval compare")
    parser.add_argument("--input", required=True, help="ask pipeline trace json path")
    parser.add_argument("--out-png", default="", help="output png path")
    parser.add_argument("--out-report", default="", help="output report json path")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    stem = input_path.stem
    out_png = Path(args.out_png).resolve() if args.out_png else input_path.with_name(f"{stem}_compare.png")
    out_report = Path(args.out_report).resolve() if args.out_report else input_path.with_name(f"{stem}_compare_report.json")

    plot(input_path, out_png, out_report)
    print(f"saved_png={out_png}")
    print(f"saved_report={out_report}")


if __name__ == "__main__":
    main()
