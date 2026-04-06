# -*- coding: utf-8 -*-
"""从 ask_result_*.json 抽出 pre-rerank、rerank 输出、final_ranked 写入 report/rerank_stages_<trace_id>.json。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = AGENT_ROOT / "report"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export pre_rerank, rerank_candidates_out, final_ranked from ask_result JSON"
    )
    parser.add_argument("ask_result", type=Path, help="Path to ask_result_<trace_id>.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: report/rerank_stages_<trace_id>.json under src/agent)",
    )
    args = parser.parse_args()
    path = args.ask_result.resolve()
    if not path.is_file():
        print(f"Missing file: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    meta_in = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    trace_id = str(meta_in.get("trace_id") or "").strip()
    if not trace_id and path.name.startswith("ask_result_") and path.suffix == ".json":
        trace_id = path.stem.removeprefix("ask_result_")
    req = data.get("request") if isinstance(data.get("request"), dict) else {}
    resp = data.get("response") if isinstance(data.get("response"), dict) else {}
    pt = resp.get("pipeline_trace") if isinstance(resp.get("pipeline_trace"), dict) else {}

    pre = pt.get("rerank_stage_pre_rerank")
    rout = pt.get("rerank_stage_rerank_out")
    final = pt.get("rerank_stage_final_ranked")
    note: str | None = None
    used_legacy_rr_hits = False

    if not isinstance(pre, list) or len(pre) == 0:
        rr = pt.get("retrieval_rerank_hits") if isinstance(pt.get("retrieval_rerank_hits"), dict) else {}
        pre = rr.get("input") if isinstance(rr.get("input"), list) else []
        used_legacy_rr_hits = True
        note = (
            "legacy: 使用 retrieval_rerank_hits.input/output，无 text_preview；"
            "重新跑 ask 后可用 rerank_stage_* 带正文预览"
        )
    if not isinstance(rout, list) or len(rout) == 0:
        rr = pt.get("retrieval_rerank_hits") if isinstance(pt.get("retrieval_rerank_hits"), dict) else {}
        fallback_out = rr.get("output") if isinstance(rr.get("output"), list) else []
        rout = fallback_out
        if note is None and fallback_out:
            note = "rerank_candidates_out 来自 retrieval_rerank_hits.output（与 rerank API 输出一致）"
    if not isinstance(final, list):
        final = []

    rerank_meta = pt.get("rerank") if isinstance(pt.get("rerank"), dict) else {}

    if not final and used_legacy_rr_hits and note:
        note = (
            f"{note}；final_ranked 在旧版 trace 中未单独落盘，"
            "重新跑 ask 后由 rerank_stage_final_ranked 提供（coverage 之后、进 LLM 前的列表）"
        )

    out_payload: dict = {
        "meta": {
            "trace_id": trace_id,
            "source_file": str(path.name),
            "question": req.get("question"),
            "export_note": note,
        },
        "rerank_summary": {
            "pre_rerank_count": len(pre),
            "rerank_candidates_in": rerank_meta.get("candidates_in"),
            "rerank_candidates_out": rerank_meta.get("candidates_out"),
            "final_ranked_count": len(final),
        },
        "pre_rerank": pre,
        "rerank_candidates_out": rout,
        "final_ranked": final,
    }

    out_path = args.output
    if out_path is None:
        out_path = REPORT_DIR / f"rerank_stages_{trace_id}.json"
    else:
        out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
