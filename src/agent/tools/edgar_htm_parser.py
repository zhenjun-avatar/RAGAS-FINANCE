"""EDGAR 10-K/10-Q HTML parser for RAG ingestion.

Two-path parsing strategy:
  - Financial tables : identified by numeric-cell density → Markdown table
  - Narrative text   : extracted from leaf <div> elements → grouped by section

The output JSON can be consumed directly by the chunking / ingestion pipeline.

Usage (standalone):
    python src/agent/tools/edgar_htm_parser.py <htm_file> [--output out.json] [--summary]
    python -m src.agent.tools.edgar_htm_parser <htm_file> --summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "edgar_htm_v1"
PARSED_JSON_SUFFIX = ".parsed.json"

# A table is "financial" when ≥30 % of its non-empty cells look numeric
# and it has at least 6 non-empty cells.
_FINANCIAL_DENSITY_THRESHOLD = 0.30
_FINANCIAL_MIN_CELLS = 6

# Narrative blocks are flushed (become a single element) when they exceed
# this character count, regardless of section boundaries.
_NARRATIVE_MAX_CHARS = 3000

# Regex helpers
_WS_RE = re.compile(r"\s+")
_NUMERIC_CELL_RE = re.compile(r"^[\s$€¥()%,.\-–—\d]+$")
_PART_RE = re.compile(r"^Part\s+(I{1,3}V?|IV|VI{0,3})\b", re.IGNORECASE)
_ITEM_RE = re.compile(r"^(Item\s+\d+[A-Z]?\.)\s*(.*)", re.IGNORECASE)

# Well-known 10-Q subsection names (lowercase) → used for section detection
_KNOWN_SUBSECTIONS: frozenset[str] = frozenset(
    {
        "financial statements",
        "management's discussion and analysis",
        "management's discussion and analysis of financial condition and results of operations",
        "quantitative and qualitative disclosures about market risk",
        "controls and procedures",
        "legal proceedings",
        "risk factors",
        "unregistered sales of equity securities and use of proceeds",
        "defaults upon senior securities",
        "mine safety disclosures",
        "other information",
        "exhibits",
    }
)

# Map (form_type, part_roman, item_num) → canonical finance_section tag.
# Keys are all lowercase; form_type strips "/a" suffix (10-q/a → 10-q).
# References: SEC Regulation S-K (17 CFR 229) and Regulation S-X.
_FINANCE_SECTION_MAP: dict[tuple[str, str, str], str] = {
    # ── 10-Q ──────────────────────────────────────────────────────────────
    # Part I – Financial Information
    ("10-q", "i",   "1"):  "financial_statements",
    ("10-q", "i",   "2"):  "mda",
    ("10-q", "i",   "3"):  "market_risk",
    ("10-q", "i",   "4"):  "controls",
    # Part II – Other Information
    ("10-q", "ii",  "1"):  "legal_proceedings",
    ("10-q", "ii",  "1a"): "risk_factors",
    ("10-q", "ii",  "2"):  "equity_sales",
    ("10-q", "ii",  "3"):  "defaults",
    ("10-q", "ii",  "4"):  "mine_safety",
    ("10-q", "ii",  "5"):  "other_info",
    ("10-q", "ii",  "6"):  "exhibits",

    # ── 10-K ──────────────────────────────────────────────────────────────
    # Part I
    ("10-k", "i",   "1"):  "business",
    ("10-k", "i",   "1a"): "risk_factors",
    ("10-k", "i",   "1b"): "unresolved_staff_comments",
    ("10-k", "i",   "1c"): "cybersecurity",          # SEC rule eff. 2023-12-15
    ("10-k", "i",   "2"):  "properties",
    ("10-k", "i",   "3"):  "legal_proceedings",
    ("10-k", "i",   "4"):  "mine_safety",
    # Part II
    ("10-k", "ii",  "5"):  "equity_market",
    ("10-k", "ii",  "6"):  "selected_financials",    # eliminated for FY ≥ 2021
    ("10-k", "ii",  "7"):  "mda",
    ("10-k", "ii",  "7a"): "market_risk",
    ("10-k", "ii",  "8"):  "financial_statements",
    ("10-k", "ii",  "9"):  "disagreements_accountants",
    ("10-k", "ii",  "9a"): "controls",
    ("10-k", "ii",  "9b"): "other_info",
    ("10-k", "ii",  "9c"): "foreign_jurisdictions",  # HFCAA eff. 2022
    # Part III
    ("10-k", "iii", "10"): "directors_governance",
    ("10-k", "iii", "11"): "executive_compensation",
    ("10-k", "iii", "12"): "security_ownership",
    ("10-k", "iii", "13"): "related_transactions",
    ("10-k", "iii", "14"): "accountant_fees",
    # Part IV
    ("10-k", "iv",  "15"): "exhibits",
    ("10-k", "iv",  "16"): "form_summary",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ParsedElement:
    element_id: str
    element_type: str  # "financial_table" | "narrative"
    section_path: list[str]
    text_for_retrieval: str
    table_name: str | None
    metadata: dict[str, Any]


@dataclass
class ParsedDocument:
    pipeline_version: str
    source_htm: str
    accession: str | None
    element_count: int
    elements: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Section tracker
# ---------------------------------------------------------------------------


class _SectionTracker:
    """Maintains a (part, item, subsection) stack as the document is walked."""

    def __init__(self) -> None:
        self._part: str | None = None
        self._item: str | None = None
        self._item_num: str | None = None   # "1", "1a", "2", …
        self._subsection: str | None = None

    @property
    def path(self) -> list[str]:
        parts = []
        if self._part:
            parts.append(self._part)
        if self._item:
            parts.append(self._item)
        if self._subsection:
            parts.append(self._subsection)
        return parts

    @property
    def item_num(self) -> str | None:
        return self._item_num

    @property
    def part_roman(self) -> str | None:
        """Return the current Part as a lowercase Roman numeral ('i', 'ii', …)."""
        if not self._part:
            return None
        m = re.search(r"(I{1,3}V?|IV|VI{0,3})$", self._part, re.IGNORECASE)
        return m.group(1).lower() if m else None

    def update(self, text: str) -> bool:
        """Try to advance the section state. Returns True on a match."""
        clean = _ws(text)
        if not clean:
            return False

        m = _PART_RE.match(clean)
        if m:
            self._part = f"Part {m.group(1).upper()}"
            self._item = self._item_num = self._subsection = None
            return True

        m = _ITEM_RE.match(clean)
        if m:
            item_label = m.group(1)                          # "Item 2."
            title = m.group(2).strip()                       # "MD&A" or ""
            num = re.search(r"\d+[A-Z]?", item_label)
            self._item = item_label
            self._item_num = num.group(0).lower() if num else None
            self._subsection = title or None
            return True

        lower = clean.lower()
        if lower in _KNOWN_SUBSECTIONS and self._item:
            self._subsection = clean
            return True

        return False

    def set_subsection(self, text: str) -> None:
        clean = _ws(text)
        if clean and len(clean) <= 150:
            self._subsection = clean


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _ws(text: str) -> str:
    """Collapse whitespace."""
    return _WS_RE.sub(" ", text).strip()


def _is_numeric_cell(text: str) -> bool:
    t = text.strip()
    return bool(t and _NUMERIC_CELL_RE.match(t))


def _classify_table(table: Tag) -> str:
    """Return 'financial' or 'layout'."""
    cells = table.find_all(["td", "th"])
    texts = [_ws(c.get_text(separator=" ")) for c in cells]
    non_empty = [t for t in texts if t]
    if len(non_empty) < _FINANCIAL_MIN_CELLS:
        return "layout"
    numeric = sum(1 for t in non_empty if _is_numeric_cell(t))
    density = numeric / len(non_empty)
    return "financial" if density >= _FINANCIAL_DENSITY_THRESHOLD else "layout"


def _table_to_markdown(table: Tag) -> str:
    """Convert a financial leaf table to a Markdown table string."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [_ws(c.get_text(separator=" ")) for c in tr.find_all(["td", "th"])]
        if any(c for c in cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "|".join("---" for _ in range(width)) + "|")
    return "\n".join(lines)


def _preceding_caption(table: Tag) -> str | None:
    """Walk backwards through siblings / parents to find a caption-like text."""
    for candidate in table.find_all_previous(["div", "p"], limit=5):
        text = _ws(candidate.get_text(separator=" "))
        if text and 8 <= len(text) <= 200 and not _NUMERIC_CELL_RE.match(text):
            return text
    return None


def _finance_section_tag(
    form_type: str | None,
    part: str | None,
    item_num: str | None,
) -> str | None:
    """Look up the canonical finance_section tag for the current position.

    Returns None when the combination is unknown (safe to omit from metadata).
    """
    if not item_num:
        return None
    key = (
        (form_type or "10-q").lower().replace("/a", ""),  # "10-q/a" → "10-q"
        (part or "").lower(),
        item_num.lower(),
    )
    return _FINANCE_SECTION_MAP.get(key)


def _looks_like_header(text: str) -> bool:
    clean = _ws(text)
    return bool(
        _PART_RE.match(clean)
        or _ITEM_RE.match(clean)
        or clean.lower() in _KNOWN_SUBSECTIONS
    )


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


class EdgarHtmParser:
    """Walk EDGAR HTML and emit ParsedElement objects in document order."""

    def __init__(
        self,
        accession: str | None = None,
        form_type: str = "10-q",
    ) -> None:
        self._accession = accession
        # Normalise: "10-Q/A" → "10-q", "10-K" → "10-k"
        self._form_type = form_type.lower().replace("/a", "")
        self._tracker = _SectionTracker()
        self._counter = 0
        # Pending narrative text fragments for the current section
        self._pending: list[str] = []
        self._pending_chars = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self, html: str) -> list[ParsedElement]:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body") or soup
        elements: list[ParsedElement] = []
        self._walk(body, elements)
        self._flush_narrative(elements)
        return elements

    # ------------------------------------------------------------------
    # DOM walker
    # ------------------------------------------------------------------

    def _walk(self, root: Tag, out: list[ParsedElement]) -> None:
        """Recursively walk *root*, collecting leaf-div text and leaf tables."""
        for node in root.children:
            if not isinstance(node, Tag):
                continue
            tag = (node.name or "").lower()

            if tag == "table":
                if node.find("table"):
                    # Container table — recurse to reach inner leaf tables
                    # but first collect any direct-cell text that lives
                    # between the nested tables
                    self._collect_direct_cell_text(node, out)
                    self._walk(node, out)
                else:
                    # Leaf table
                    self._handle_leaf_table(node, out)

            elif tag == "div":
                if node.find(["table", "div"]):
                    # Compound div — recurse
                    self._walk(node, out)
                elif node.find("a", attrs={"name": True}):
                    # Workiva wraps jump targets as <div><a name="…"></a></div>.
                    # That outer div has no nested div/table, so it would be misclassified
                    # as a "leaf" and we'd skip children — never seeing <a name> flushes.
                    self._walk(node, out)
                else:
                    # Leaf div — the atomic text unit in EDGAR Wdesk HTML
                    text = _ws(node.get_text(separator=" "))
                    if text:
                        self._handle_text(text, out)

            elif tag == "a" and node.get("name"):
                # Named anchor = section jump target in EDGAR (Workiva) HTML.
                # Flush any pending narrative so the upcoming content starts
                # a fresh element with the correct section_path.
                self._flush_narrative(out)

            elif tag in ("td", "th"):
                # Walk into cells that may contain compound content
                if node.find(["table", "div"]):
                    self._walk(node, out)
                else:
                    text = _ws(node.get_text(separator=" "))
                    if text:
                        self._handle_text(text, out)

            elif tag in ("tr", "tbody", "thead", "tfoot", "section",
                         "article", "main", "header", "footer"):
                self._walk(node, out)

    def _collect_direct_cell_text(self, table: Tag, out: list[ParsedElement]) -> None:
        """Extract text from cells that contain NO nested tables."""
        for td in table.find_all(["td", "th"], recursive=True):
            if td.find("table"):
                continue  # cell has nested table; handled via _walk
            text = _ws(td.get_text(separator=" "))
            if text:
                self._handle_text(text, out)

    # ------------------------------------------------------------------
    # Text & table handlers
    # ------------------------------------------------------------------

    def _handle_text(self, text: str, out: list[ParsedElement]) -> None:
        if _looks_like_header(text):
            self._flush_narrative(out)
            self._tracker.update(text)
            return

        # Accumulate narrative; flush if section would overflow
        if self._pending_chars + len(text) > _NARRATIVE_MAX_CHARS and self._pending:
            self._flush_narrative(out)

        self._pending.append(text)
        self._pending_chars += len(text)

    def _handle_leaf_table(self, table: Tag, out: list[ParsedElement]) -> None:
        kind = _classify_table(table)
        if kind == "financial":
            # Flush any pending narrative first (preserve document order)
            self._flush_narrative(out)
            md = _table_to_markdown(table)
            if not md:
                return
            caption = _preceding_caption(table)
            text = f"## {caption}\n\n{md}" if caption else md
            out.append(
                ParsedElement(
                    element_id=self._next_id("tbl"),
                    element_type="financial_table",
                    section_path=list(self._tracker.path),
                    text_for_retrieval=text,
                    table_name=caption,
                    metadata=self._meta("financial_table"),
                )
            )
        else:
            # Treat layout table cells as narrative text
            for tr in table.find_all("tr"):
                cells = [
                    _ws(c.get_text(separator=" "))
                    for c in tr.find_all(["td", "th"])
                ]
                row_text = " ".join(c for c in cells if c)
                if row_text:
                    self._handle_text(row_text, out)

    # ------------------------------------------------------------------
    # Narrative flush
    # ------------------------------------------------------------------

    def _flush_narrative(self, out: list[ParsedElement]) -> None:
        if not self._pending:
            return
        text = "\n\n".join(p for p in self._pending if p).strip()
        self._pending = []
        self._pending_chars = 0
        if not text or len(text) < 20:
            return
        out.append(
            ParsedElement(
                element_id=self._next_id("nar"),
                element_type="narrative",
                section_path=list(self._tracker.path),
                text_for_retrieval=text,
                table_name=None,
                metadata=self._meta("narrative"),
            )
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:04d}"

    def _meta(self, content_type: str) -> dict[str, Any]:
        meta: dict[str, Any] = {"content_type": content_type}
        fs = _finance_section_tag(
            self._form_type,
            self._tracker.part_roman,
            self._tracker.item_num,
        )
        if fs:
            meta["finance_section"] = fs
        if self._accession:
            meta["finance_accns"] = [self._accession]
        return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_edgar_htm(
    htm_path: Path,
    *,
    accession: str | None = None,
    form_type: str = "10-q",
    force: bool = False,
    output_path: Path | None = None,
) -> tuple[list[ParsedElement], Path]:
    """Parse *htm_path* and write a ``*.parsed.json`` sidecar.

    Returns ``(elements, json_path)``.  The JSON is re-used on subsequent
    calls unless *force* is True.
    """
    json_path = output_path or htm_path.with_name(htm_path.stem + PARSED_JSON_SUFFIX)

    if (
        not force
        and json_path.is_file()
        and json_path.stat().st_mtime >= htm_path.stat().st_mtime
    ):
        logger.info("[edgar_parser] Cache hit → {}", json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return [_dict_to_element(d) for d in data.get("elements", [])], json_path

    # Infer accession from EDGAR filename pattern: EDGAR_{cik}_{accession}.htm
    if accession is None:
        parts = htm_path.stem.split("_", 2)
        accession = parts[2] if len(parts) >= 3 else None

    raw = htm_path.read_text(encoding="utf-8", errors="replace")
    # Strip the EDGAR SGML envelope (<DOCUMENT> … <TEXT>) if present
    html_start = raw.lower().find("<html")
    if html_start != -1:
        raw = raw[html_start:]

    parser = EdgarHtmParser(accession=accession, form_type=form_type)
    elements = parser.parse(raw)

    doc = ParsedDocument(
        pipeline_version=PIPELINE_VERSION,
        source_htm=str(htm_path.resolve()),
        accession=accession,
        element_count=len(elements),
        elements=[asdict(e) for e in elements],
    )
    json_path.write_text(
        json.dumps(asdict(doc), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "[edgar_parser] {} elements ({} tables, {} narrative) → {}",
        len(elements),
        sum(1 for e in elements if e.element_type == "financial_table"),
        sum(1 for e in elements if e.element_type == "narrative"),
        json_path.name,
    )
    return elements, json_path


def _dict_to_element(d: dict[str, Any]) -> ParsedElement:
    return ParsedElement(
        element_id=d.get("element_id", ""),
        element_type=d.get("element_type", ""),
        section_path=d.get("section_path", []),
        text_for_retrieval=d.get("text_for_retrieval", ""),
        table_name=d.get("table_name"),
        metadata=d.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edgar_htm_parser",
        description="Parse EDGAR HTML filings into structured JSON for RAG ingestion.",
    )
    p.add_argument("htm_file", help="Path to the EDGAR .htm file")
    p.add_argument(
        "--output", "-o",
        help="Output JSON path (default: <stem>.parsed.json next to input)",
    )
    p.add_argument(
        "--accession", "-a",
        help="SEC accession number (inferred from filename if omitted)",
    )
    p.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-parse even if a cached JSON is up to date",
    )
    p.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Print a human-readable summary of parsed elements",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _cli_parser().parse_args(argv)
    htm_path = Path(args.htm_file)
    if not htm_path.exists():
        print(f"Error: file not found: {htm_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    elements, json_path = parse_edgar_htm(
        htm_path,
        accession=args.accession,
        force=args.force,
        output_path=output_path,
    )

    print(f"Parsed {len(elements)} elements → {json_path}")

    if args.summary:
        from collections import Counter

        type_counts = Counter(e.element_type for e in elements)
        section_counts: Counter[str] = Counter()
        for e in elements:
            if e.section_path:
                section_counts[" > ".join(e.section_path[:2])] += 1

        print(f"\nElement types : {dict(type_counts)}")
        print(f"\nTop sections  :")
        for sec, cnt in section_counts.most_common(10):
            print(f"  {cnt:3d}  {sec}")

        print(f"\nFirst 20 elements:")
        for e in elements[:20]:
            path_str = " > ".join(e.section_path) if e.section_path else "(no section)"
            preview = e.text_for_retrieval[:70].replace("\n", " ")
            print(f"  [{e.element_id}] {e.element_type:15s} | {path_str[:40]:40s} | {preview}")


if __name__ == "__main__":
    main()
