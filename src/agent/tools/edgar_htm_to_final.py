"""EDGAR .htm → parse (BS4+rules) → enrich (rules+optional LLM) → one final JSON.

Designed to be run from the **agent** directory::

    cd src/agent
    python tools/edgar_htm_to_final.py tools/data/EDGAR_320193_0000320193-17-000009.htm
    python tools/edgar_htm_to_final.py tools/data/EDGAR_....htm -o tools/data/out.final.json
    python tools/edgar_htm_to_final.py tools/data/EDGAR_....htm --no-llm --summary

From repo root::

    python src/agent/tools/edgar_htm_to_final.py src/agent/tools/data/EDGAR_....htm
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger

if __name__ == "__main__" and not __package__:
    # ``src`` → ``from agent.tools…`` ; ``src/agent`` → ``from core.config…`` (llm.py)
    _agent_root = Path(__file__).resolve().parents[1]
    _src_root = _agent_root.parent
    for _root in (_src_root, _agent_root):
        _s = str(_root)
        if _s not in sys.path:
            sys.path.insert(0, _s)

if __package__:
    from .edgar_htm_enricher import EdgarEnricher
    from .edgar_htm_parser import (
        PIPELINE_VERSION as PARSER_VERSION,
        EdgarHtmParser,
        ParsedDocument,
        ParsedElement,
    )
else:
    from agent.tools.edgar_htm_enricher import EdgarEnricher
    from agent.tools.edgar_htm_parser import (
        PIPELINE_VERSION as PARSER_VERSION,
        EdgarHtmParser,
        ParsedDocument,
        ParsedElement,
    )

FINAL_JSON_SUFFIX = ".final.json"


def _infer_accession(htm_path: Path) -> str | None:
    parts = htm_path.stem.split("_", 2)
    return parts[2] if len(parts) >= 3 else None


def parse_htm_to_elements(
    htm_path: Path,
    *,
    accession: str | None = None,
    form_type: str = "10-q",
) -> list[ParsedElement]:
    """Read *htm_path* and return parsed elements (no JSON written)."""
    accession = accession if accession is not None else _infer_accession(htm_path)
    raw = htm_path.read_text(encoding="utf-8", errors="replace")
    html_start = raw.lower().find("<html")
    if html_start != -1:
        raw = raw[html_start:]
    return EdgarHtmParser(accession=accession, form_type=form_type).parse(raw)


def htm_to_final_json(
    htm_path: Path,
    *,
    output_path: Path | None = None,
    accession: str | None = None,
    form_type: str = "10-q",
    use_llm: bool = True,
    write_parsed: bool = False,
) -> tuple[list[ParsedElement], Path]:
    """Parse *htm_path*, enrich, write ``<stem>.final.json`` (or *output_path*)."""
    htm_path = htm_path.resolve()
    elements = parse_htm_to_elements(htm_path, accession=accession, form_type=form_type)
    accession = accession if accession is not None else _infer_accession(htm_path)

    if write_parsed:
        parsed_path = htm_path.with_name(htm_path.stem + ".parsed.json")
        doc = ParsedDocument(
            pipeline_version=PARSER_VERSION,
            source_htm=str(htm_path),
            accession=accession,
            element_count=len(elements),
            elements=[asdict(e) for e in elements],
        )
        parsed_path.write_text(
            json.dumps(asdict(doc), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("[pipeline] wrote intermediate {}", parsed_path.name)

    enriched = EdgarEnricher(use_llm=use_llm).enrich(elements)

    out = output_path or htm_path.with_name(htm_path.stem + FINAL_JSON_SUFFIX)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "pipeline_version": f"{PARSER_VERSION}+enriched",
        "source_htm": str(htm_path),
        "accession": accession,
        "element_count": len(enriched),
        "elements": [asdict(e) for e in enriched],
    }
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "[pipeline] {} elements → {}",
        len(enriched),
        out.name,
    )
    return enriched, out


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EDGAR .htm → parsed + enriched JSON (run from src/agent).",
    )
    p.add_argument("htm_file", type=Path, help="Path to .htm (e.g. tools/data/EDGAR_....htm)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON (default: <stem>{FINAL_JSON_SUFFIX} next to .htm)",
    )
    p.add_argument(
        "-a",
        "--accession",
        default=None,
        help="SEC accession (default: from EDGAR_*_*_accession.htm filename)",
    )
    p.add_argument(
        "--form-type",
        default="10-q",
        metavar="TYPE",
        help="SEC form type: 10-q, 10-k, 10-q/a, 10-k/a … (default: 10-q)",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Enrich with rules only (no LLM calls)",
    )
    p.add_argument(
        "--write-parsed",
        action="store_true",
        help="Also write <stem>.parsed.json beside the .htm",
    )
    p.add_argument("--summary", action="store_true", help="Print counts after run")
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
        print(f"Error: not found: {htm}", file=sys.stderr)
        return 1

    try:
        enriched, out_path = htm_to_final_json(
            htm,
            output_path=args.output,
            accession=args.accession,
            form_type=args.form_type,
            use_llm=not args.no_llm,
            write_parsed=args.write_parsed,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Final: {len(enriched)} elements → {out_path}")

    if args.summary:
        from collections import Counter

        print(f"  types: {dict(Counter(e.element_type for e in enriched))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
