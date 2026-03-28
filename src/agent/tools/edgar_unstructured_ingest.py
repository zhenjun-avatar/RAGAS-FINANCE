"""EDGAR .htm → Unstructured partition → sidecar JSON → plain text for ingestion.

Writes ``<stem>.unstructured.json`` next to the .htm. Table elements use
``text_as_html`` → row-wise ``TABLE | col | …`` in ``text_structured``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

PIPELINE_VERSION = "edgar_unstructured_v1"
JSON_SUFFIX = ".unstructured.json"


def extract_inner_html(raw: str) -> str:
    lower = raw.lower()
    start = lower.find("<html")
    if start == -1:
        return raw
    end = lower.rfind("</html>")
    if end == -1:
        return raw[start:]
    return raw[start : end + len("</html>")]


def unstructured_json_path(htm_path: Path) -> Path:
    return htm_path.with_name(htm_path.stem + JSON_SUFFIX)


def table_rows_from_html(table_html: str) -> list[str]:
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
        cells = [c.get_text(separator=" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not any(c.strip() for c in cells):
            continue
        rows.append("TABLE | " + " | ".join(cells))
    return rows


def structured_table_text_from_element(el: object) -> str | None:
    cat = (getattr(el, "category", None) or "").lower()
    if cat != "table":
        return None
    meta = getattr(el, "metadata", None)
    if meta is None:
        return None
    html_snippet = getattr(meta, "text_as_html", None)
    if not html_snippet:
        to_dict = getattr(meta, "to_dict", None)
        if callable(to_dict):
            d = to_dict()
            html_snippet = d.get("text_as_html") if isinstance(d, dict) else None
    if not html_snippet:
        return None
    lines = table_rows_from_html(str(html_snippet))
    return "\n".join(lines) if lines else None


def serialize_element(el: object) -> dict[str, Any]:
    category = str(getattr(el, "category", None) or type(el).__name__)
    text = getattr(el, "text", None) or ""
    row: dict[str, Any] = {"category": category, "text": text}
    st = structured_table_text_from_element(el)
    if st:
        row["text_structured"] = st
    meta = getattr(el, "metadata", None)
    if meta is not None:
        to_dict = getattr(meta, "to_dict", None)
        if callable(to_dict):
            row["metadata"] = to_dict()
        elif isinstance(meta, dict):
            row["metadata"] = meta
    return row


def elements_dicts_to_body_text(elements: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for e in elements:
        ts = e.get("text_structured")
        if isinstance(ts, str) and ts.strip():
            parts.append(ts.strip())
            continue
        t = (e.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def partition_htm_to_elements(htm_path: Path) -> list[Any]:
    from unstructured.partition.html import partition_html

    raw = htm_path.read_text(encoding="utf-8", errors="replace")
    html = extract_inner_html(raw)
    return list(partition_html(text=html))


def write_unstructured_json(htm_path: Path, elements: list[Any]) -> Path:
    json_path = unstructured_json_path(htm_path)
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "source_htm": str(htm_path.resolve()),
        "element_count": len(elements),
        "elements": [serialize_element(el) for el in elements],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("[EDGAR] Wrote Unstructured sidecar {}", json_path)
    return json_path


def load_unstructured_json(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def ensure_unstructured_json(htm_path: Path, *, force: bool = False) -> Path:
    """Partition if missing or stale; return path to sidecar JSON."""
    json_path = unstructured_json_path(htm_path)
    if (
        not force
        and json_path.is_file()
        and json_path.stat().st_mtime >= htm_path.stat().st_mtime
    ):
        return json_path
    elements = partition_htm_to_elements(htm_path)
    return write_unstructured_json(htm_path, elements)


def body_text_from_unstructured_json(json_path: Path) -> str:
    data = load_unstructured_json(json_path)
    elements = data.get("elements") or []
    if not isinstance(elements, list):
        return ""
    return elements_dicts_to_body_text(elements)


def prepare_edgar_htm_with_unstructured(htm_path: Path, *, force_json: bool = False) -> tuple[str, Path, dict[str, Any]]:
    """
    Returns (body_text, json_path, sidecar_meta) for embedding/chunking.
    Caller prepends SEC header if needed.
    """
    json_path = ensure_unstructured_json(htm_path, force=force_json)
    body = body_text_from_unstructured_json(json_path)
    meta = {
        "parser": "unstructured",
        "unstructured_json": str(json_path.resolve()),
        "table_row_count": body.count("\nTABLE | ") + (1 if body.startswith("TABLE | ") else 0),
    }
    return body, json_path, meta
