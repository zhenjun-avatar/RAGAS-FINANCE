"""Experiment: partition any .htm with unstructured.io ``partition_html``.

Writes JSON compatible with ``edgar_unstructured_ingest.serialize_element`` shape,
plus extra stats for comparing against ``edgar_htm_parser.py`` output.

Requires::

    pip install unstructured

Run (from repo root)::

    python src/agent/tools/unstructured_htm_experiment.py src/agent/tools/data/EDGAR_320193_0000320193-17-000009.htm
    python src/agent/tools/unstructured_htm_experiment.py path/to/file.htm -o logs/u.json --summary --body-txt

Or with ``PYTHONPATH=src``::

    python -m agent.tools.unstructured_htm_experiment src/agent/tools/data/EDGAR_....htm --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Allow ``python src/agent/tools/unstructured_htm_experiment.py`` (no package context).
if __name__ == "__main__" and not __package__:
    _agent_root = Path(__file__).resolve().parents[1]
    _src_root = _agent_root.parent
    for _root in (_src_root, _agent_root):
        _s = str(_root)
        if _s not in sys.path:
            sys.path.insert(0, _s)

from agent.tools.edgar_unstructured_ingest import (
    elements_dicts_to_body_text,
    extract_inner_html,
    serialize_element,
)

from loguru import logger

EXPERIMENT_PIPELINE_VERSION = "edgar_unstructured_experiment_v1"
DEFAULT_JSON_SUFFIX = ".unstructured.experiment.json"


def _unstructured_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("unstructured")
    except Exception:
        return None


def partition_htm_file(htm_path: Path) -> list[Any]:
    from unstructured.partition.html import partition_html

    raw = htm_path.read_text(encoding="utf-8", errors="replace")
    html = extract_inner_html(raw)
    return list(partition_html(text=html))


def run_experiment(
    htm_path: Path,
    *,
    output_json: Path | None = None,
    body_txt_path: Path | None = None,
) -> dict[str, Any]:
    elements = partition_htm_file(htm_path)
    serialized = [serialize_element(el) for el in elements]
    cats = Counter((e.get("category") or "?") for e in serialized)
    text_lens = [len((e.get("text") or "")) for e in serialized]
    payload: dict[str, Any] = {
        "pipeline_version": EXPERIMENT_PIPELINE_VERSION,
        "unstructured_library_version": _unstructured_version(),
        "source_htm": str(htm_path.resolve()),
        "element_count": len(serialized),
        "category_histogram": dict(cats.most_common()),
        "total_text_chars": sum(text_lens),
        "max_element_text_chars": max(text_lens) if text_lens else 0,
        "elements": serialized,
    }
    out = output_json or htm_path.with_name(htm_path.stem + DEFAULT_JSON_SUFFIX)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("[unstructured_experiment] {} elements → {}", len(serialized), out)

    if body_txt_path is not None:
        body = elements_dicts_to_body_text(serialized)
        body_txt_path.parent.mkdir(parents=True, exist_ok=True)
        body_txt_path.write_text(body, encoding="utf-8")
        logger.info("[unstructured_experiment] body text → {} chars → {}", len(body), body_txt_path)

    return payload


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment: unstructured.io partition_html on .htm files.",
    )
    p.add_argument("htm_file", type=Path, help="Path to .htm")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON (default: <stem>{DEFAULT_JSON_SUFFIX})",
    )
    p.add_argument(
        "--body-txt",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write concatenated body text (same merge as edgar_unstructured_ingest)",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print category histogram and first few elements",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = _cli().parse_args(argv)
    htm = args.htm_file
    if not htm.is_file():
        print(f"Error: not a file: {htm}", file=sys.stderr)
        return 1

    try:
        payload = run_experiment(htm, output_json=args.output, body_txt_path=args.body_txt)
    except ImportError:
        print(
            "Missing dependency. Install: python -m pip install unstructured",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3

    out_path = args.output or htm.with_name(htm.stem + DEFAULT_JSON_SUFFIX)
    print(f"Wrote {payload['element_count']} elements → {out_path.resolve()}")

    if args.summary:
        print(f"unstructured version: {payload.get('unstructured_library_version')}")
        print(f"total text chars (sum of element.text): {payload['total_text_chars']}")
        print("categories:", payload["category_histogram"])
        print("\nFirst 15 elements (category, text[:120]):")
        for i, el in enumerate(payload["elements"][:15]):
            cat = el.get("category", "?")
            t = (el.get("text") or "").replace("\n", " ")[:120]
            print(f"  [{i:02d}] {cat}: {t}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
