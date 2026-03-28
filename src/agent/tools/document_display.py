"""Human-readable document labels. Stable identity remains `document_id` only."""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Priority for display_name (extend in _resolve_display_fields):
# 1) entity + form + filed (+ accession if not redundant)
# 2) EDGAR · accession (from metadata, title, or EDGAR_{cik}_{accession}.htm)
# 3) non-empty title / raw basename
# 4) generic "Document {id}"

_EDGAR_FILE = re.compile(
    r"^EDGAR_(\d+)_(\d{10}-\d{2}-\d{6})(?:\.[a-z0-9]+)?$",
    re.IGNORECASE,
)
_EDGAR_TITLE = re.compile(r"^EDGAR\s+(.+?)\s*\|\s*(.+)$", re.IGNORECASE)


def parse_edgar_filename(stem_or_path: str | None) -> tuple[int | None, str | None]:
    """Parse ``EDGAR_{cik}_{accession}.htm`` (or similar) basename."""
    if not stem_or_path:
        return None, None
    base = os.path.basename(stem_or_path.strip())
    m = _EDGAR_FILE.match(base)
    if not m:
        return None, None
    try:
        cik = int(m.group(1))
    except ValueError:
        cik = None
    return cik, m.group(2)


def parse_edgar_title_line(title: str | None) -> tuple[str | None, str | None]:
    """Parse ingestion title like ``EDGAR 10-K | 0001193125-20-310684``."""
    t = (title or "").strip()
    if not t:
        return None, None
    m = _EDGAR_TITLE.match(t)
    if not m:
        return None, None
    form_raw = m.group(1).strip()
    accn = m.group(2).strip()
    if form_raw in ("?", ""):
        form_raw = ""
    return form_raw or None, accn or None


def metadata_as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _resolve_display_fields(
    *,
    title: str | None,
    source_uri: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge DB metadata with hints from EDGAR title line and filename."""
    meta = dict(metadata)
    t_form, t_accn = parse_edgar_title_line(title)
    f_cik, f_accn = parse_edgar_filename(source_uri or title)
    accn = str(meta.get("sec_accession") or meta.get("accn") or "").strip() or t_accn or f_accn or ""
    form = str(meta.get("form") or meta.get("sec_form") or "").strip() or (t_form or "")
    filed = str(meta.get("filed") or meta.get("sec_filing_date") or "").strip()
    entity = str(meta.get("entity_name") or meta.get("entityName") or "").strip()
    cik = meta.get("cik")
    if cik is None and f_cik is not None:
        cik = f_cik
    return {
        "entity_name": entity,
        "form": form,
        "filed": filed,
        "accn": accn,
        "cik": cik,
    }


def _subtitle_for_catalog(document_id: int, file_type: str | None, accn: str) -> str:
    parts = [f"ID {document_id}"]
    if file_type:
        parts.append(str(file_type))
    if accn:
        parts.append(accn)
    return " · ".join(parts)


def _display_name_from_fields(
    *,
    document_id: int,
    title: str | None,
    source_uri: str | None,
    fields: dict[str, Any],
) -> str:
    entity = fields["entity_name"]
    form = fields["form"]
    filed = fields["filed"]
    accn = fields["accn"]
    base_parts = [p for p in (entity, form, filed) if p]
    if base_parts:
        label = " · ".join(base_parts)
        if accn and accn not in label:
            label += f" · {accn}"
        return label
    if accn:
        return f"EDGAR · {accn}"
    raw_name = os.path.basename((source_uri or title or "").strip())
    if raw_name and raw_name not in (".", ".."):
        return raw_name
    fallback_title = str(title or "").strip()
    if fallback_title:
        return fallback_title
    return f"Document {document_id}"


def _raw_filename_hint(
    *,
    title: str | None,
    source_uri: str | None,
    metadata: dict[str, Any],
) -> str | None:
    for key in ("primary_document", "file_name", "filename"):
        v = metadata.get(key)
        if v:
            b = os.path.basename(str(v).strip())
            if b and b not in (".", ".."):
                return b
    base = os.path.basename((source_uri or title or "").strip())
    if base and base not in (".", ".."):
        return base
    return None


def build_document_catalog_row(
    *,
    document_id: int,
    title: str | None,
    file_type: str | None,
    metadata: dict[str, Any] | None,
    source_uri: str | None = None,
) -> dict[str, Any]:
    """Single pass: labels + resolved fields for ``GET /documents/catalog``."""
    meta = metadata or {}
    fields = _resolve_display_fields(title=title, source_uri=source_uri, metadata=meta)
    display_name = _display_name_from_fields(
        document_id=document_id, title=title, source_uri=source_uri, fields=fields
    )
    accn = fields["accn"]
    subtitle = _subtitle_for_catalog(document_id, file_type, accn)
    raw_filename = _raw_filename_hint(title=title, source_uri=source_uri, metadata=meta)
    return {
        "display_name": display_name,
        "subtitle": subtitle,
        "entity_name": fields["entity_name"] or None,
        "form": fields["form"] or None,
        "filed": fields["filed"] or None,
        "accn": accn or None,
        "cik": fields.get("cik"),
        "raw_filename": raw_filename,
    }
