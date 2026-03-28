#!/usr/bin/env python3
"""
Standalone: parse an EDGAR .htm with Unstructured (partition_html).

Not integrated with ingestion. Install once::

    cd src/agent
    ..\\venv\\Scripts\\python.exe -m pip install "unstructured[html]"

Run::

    ..\\venv\\Scripts\\python.exe scripts\\run_unstructured_edgar_html.py tools\\data\\EDGAR_1823776_0001193125-21-100318.htm
    ..\\venv\\Scripts\\python.exe scripts\\run_unstructured_edgar_html.py tools\\data\\EDGAR_1823776_0001193125-21-100318.htm --out-json logs\\unstructured_elements.json
    ..\\venv\\Scripts\\python.exe scripts\\run_unstructured_edgar_html.py tools\\data\\EDGAR_....htm --structural-tables --out-json logs\\unstructured_elements.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_inner_html(raw: str) -> str:
    """EDGAR often wraps real markup in <DOCUMENT>…<TEXT>…<HTML>…</HTML>."""
    lower = raw.lower()
    start = lower.find("<html")
    if start == -1:
        return raw
    end = lower.rfind("</html>")
    if end == -1:
        return raw[start:]
    return raw[start : end + len("</html>")]


def _table_rows_from_html(table_html: str) -> list[str]:
    """Recover row/column boundaries from Unstructured's metadata.text_as_html."""
    if not (table_html or "").strip():
        return []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if not table:
        return []
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for cell in tr.find_all(["td", "th"]):
            t = cell.get_text(separator=" ", strip=True)
            cells.append(t)
        if not any(c.strip() for c in cells):
            continue
        rows.append("TABLE | " + " | ".join(cells))
    return rows


def _structured_table_text(el: object) -> str | None:
    """For category Table, prefer HTML grid in metadata over flat el.text."""
    cat = (getattr(el, "category", None) or "").lower()
    if cat != "table":
        return None
    meta = getattr(el, "metadata", None)
    if meta is None:
        return None
    html_snippet = getattr(meta, "text_as_html", None)
    if not html_snippet:
        md = getattr(meta, "to_dict", None)
        if callable(md):
            d = md()
            html_snippet = d.get("text_as_html") if isinstance(d, dict) else None
    if not html_snippet:
        return None
    lines = _table_rows_from_html(str(html_snippet))
    return "\n".join(lines) if lines else None


def _element_summary(el: object) -> dict:
    category = getattr(el, "category", None) or type(el).__name__
    text = getattr(el, "text", None) or ""
    row: dict = {"category": str(category), "text_len": len(text), "text_preview": text[:400]}
    meta = getattr(el, "metadata", None)
    if meta is not None:
        to_dict = getattr(meta, "to_dict", None)
        if callable(to_dict):
            row["metadata"] = to_dict()
        elif isinstance(meta, dict):
            row["metadata"] = meta
        else:
            row["metadata"] = str(meta)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Partition EDGAR HTML with unstructured.io")
    parser.add_argument(
        "htm_path",
        type=Path,
        help="Path to .htm file (e.g. tools/data/EDGAR_....htm)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write all elements as JSON (can be large)",
    )
    parser.add_argument(
        "--out-text",
        type=Path,
        default=None,
        help="Write concatenated element texts (newline-separated)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Print first N elements to stdout (default 20)",
    )
    parser.add_argument(
        "--structural-tables",
        action="store_true",
        help="For Table elements, derive row/column text from metadata.text_as_html (JSON + --out-text)",
    )
    args = parser.parse_args()

    path = args.htm_path.resolve()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    try:
        from unstructured.partition.html import partition_html
    except ImportError:
        print(
            "Missing package. Install: python -m pip install \"unstructured[html]\"",
            file=sys.stderr,
        )
        return 2

    raw = path.read_text(encoding="utf-8", errors="replace")
    html = _extract_inner_html(raw)

    elements = partition_html(text=html)

    summaries = [_element_summary(el) for el in elements]

    print(f"file: {path}")
    print(f"elements: {len(elements)}")
    n_prev = max(0, args.preview)
    for i, (el, s) in enumerate(zip(elements[:n_prev], summaries[:n_prev])):
        print(f"\n--- [{i}] {s['category']} (len={s['text_len']}) ---")
        st = _structured_table_text(el) if args.structural_tables else None
        if st:
            print(st[:800] + ("…" if len(st) > 800 else ""))
        else:
            print(s.get("text_preview", "") or "(empty)")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        # Full text in JSON (large)
        full = []
        for el in elements:
            d = _element_summary(el)
            d["text"] = getattr(el, "text", None) or ""
            if args.structural_tables:
                st = _structured_table_text(el)
                if st:
                    d["text_structured"] = st
            full.append(d)
        args.out_json.write_text(
            json.dumps(
                {"source": str(path), "element_count": len(full), "elements": full},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {args.out_json}")

    if args.out_text:
        args.out_text.parent.mkdir(parents=True, exist_ok=True)
        parts = []
        for el in elements:
            if args.structural_tables:
                st = _structured_table_text(el)
                if st:
                    parts.append(st)
                    continue
            t = (getattr(el, "text", None) or "").strip()
            if t:
                parts.append(t)
        args.out_text.write_text("\n\n".join(parts), encoding="utf-8")
        print(f"Wrote text: {args.out_text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
