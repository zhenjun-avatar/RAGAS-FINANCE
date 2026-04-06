"""Post-parse enrichment for EDGAR HTML parsed elements.

BeautifulSoup + rules (edgar_htm_parser.py) handles the overall DOM structure.
This module handles the three categories of detail that rules alone handle poorly:

  1. Page headers / footers  — rules first, LLM for misses
  2. Footnote linkage        — rules extract (N) markers; LLM resolves ambiguous targets
  3. Sub-section boundaries  — LLM detects implicit titles inside long narratives

The enricher operates on the ``parsed.json`` emitted by ``edgar_htm_parser``.
It adds / modifies fields **without** changing the overall element schema, so the
result is a drop-in replacement for the original ``parsed.json``.

Usage (standalone)::

    python src/agent/tools/edgar_htm_enricher.py src/agent/tools/data/EDGAR_....parsed.json
    python src/agent/tools/edgar_htm_enricher.py src/agent/tools/data/EDGAR_....parsed.json --summary
    python src/agent/tools/edgar_htm_enricher.py src/agent/tools/data/EDGAR_....parsed.json -o enriched.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger

if __name__ == "__main__" and not __package__:
    _agent_root = Path(__file__).resolve().parents[1]
    _src_root = _agent_root.parent
    for _root in (_src_root, _agent_root):
        _s = str(_root)
        if _s not in sys.path:
            sys.path.insert(0, _s)

if __package__:
    from .edgar_htm_parser import ParsedElement, _dict_to_element
else:
    from agent.tools.edgar_htm_parser import ParsedElement, _dict_to_element

ENRICHED_JSON_SUFFIX = ".enriched.json"

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

# Page header / footer: "Apple Inc. | Q3 2017 Form 10-Q | 15"
_PAGE_STAMP_RE = re.compile(
    r"(?:^|\n)"
    r"[A-Za-z][^|\n]{1,60}"
    r"\|\s*"
    r"(?:Q[1-4]\s+\d{4}\s+)?Form\s+10-[KkQq][/A]?"
    r"\s*\|\s*\d{1,3}"
    r"(?:\n|$)",
    re.MULTILINE,
)

# Standalone noise lines that survive the pipe-stamp filter:
#   "Apple Inc."  /  "See accompanying Notes to …"  /  "Form 10-Q"
_STANDALONE_NOISE_RE = re.compile(
    r"^(?:Apple\s+Inc\.|See\s+accompanying\s+Notes\s+to\b[^\n]{0,100}|Form\s+10-[KkQq][/A]?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Footnote-only paragraph: "(1) Some text…"
_FOOTNOTE_LEADER_RE = re.compile(r"^\s*\((\d+)\)\s+(.+)", re.DOTALL)

# Short title-case line that looks like an implicit sub-heading
_IMPLICIT_HEADING_RE = re.compile(r"^([A-Z][A-Za-z /\-]{3,80})$")

# Minimum chars to keep a narrative element (below → discard as fragment)
_MIN_NARRATIVE_CHARS = 40

# LLM batch sizes
_CAPTION_BATCH = 5       # tables per LLM call for caption extraction
_SUBHEAD_MIN_CHARS = 800  # only split narratives longer than this

# A table caption is "bad" when it matches any of these conditions
def _bad_caption(name: str | None) -> bool:
    if not name:
        return True
    return (
        name.startswith("(")       # unit note e.g. "(In millions, …)"
        or "|" in name             # page-header residue
        or len(name) > 80          # too long to be a title
        or len(name) < 5           # too short
    )


def _looks_like_table_header(text: str) -> bool:
    """Return True when *text* is likely a dangling table title or unit note.

    Two patterns qualify:
    * ALL-CAPS title  — "CONDENSED CONSOLIDATED BALANCE SHEETS"
    * Unit note       — "(In millions, except …)"
    """
    s = text.strip()
    if not s or len(s) > 150:
        return False
    if re.match(r"^\(In\s+[a-zA-Z]", s):
        return True
    # ALL CAPS: every alphabetic character is uppercase, has ≥2 words
    words = s.split()
    if len(words) >= 2 and all(w == w.upper() for w in words if w.isalpha()):
        return True
    return False


# ---------------------------------------------------------------------------
# Rules helpers
# ---------------------------------------------------------------------------

def _strip_page_stamps(text: str) -> tuple[str, int]:
    """Remove page-header/footer and standalone noise lines from narrative text.

    Returns (cleaned_text, count_removed).
    """
    n_before = len(_PAGE_STAMP_RE.findall(text)) + len(_STANDALONE_NOISE_RE.findall(text))
    cleaned = _PAGE_STAMP_RE.sub("\n", text)
    cleaned = _STANDALONE_NOISE_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, n_before


def _split_implicit_headings(text: str) -> list[tuple[str | None, str]]:
    """Split text on standalone title-case lines that look like sub-headings.

    Returns list of (heading | None, body_text) pairs.
    """
    paragraphs = re.split(r"\n\n+", text.strip())
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_body: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if _IMPLICIT_HEADING_RE.match(para) and len(para.split()) <= 10:
            if current_body:
                sections.append((current_heading, "\n\n".join(current_body)))
                current_body = []
            current_heading = para
        else:
            current_body.append(para)

    if current_body:
        sections.append((current_heading, "\n\n".join(current_body)))

    return sections if sections else [(None, text)]


def _extract_footnotes(
    elements: list[ParsedElement],
) -> tuple[list[ParsedElement], dict[int, str]]:
    """Separate footnote-only narrative elements from the main list.

    Returns (remaining_elements, {footnote_number: text}).
    Footnote elements are small narratives whose *entire* content is one or
    more ``(N) …`` sentences and nothing else.
    """
    remaining: list[ParsedElement] = []
    footnotes: dict[int, str] = {}

    for el in elements:
        if el.element_type != "narrative":
            remaining.append(el)
            continue

        paras = [p.strip() for p in el.text_for_retrieval.split("\n\n") if p.strip()]
        all_footnote = all(_FOOTNOTE_LEADER_RE.match(p) for p in paras)

        if all_footnote and len(paras) <= 8:
            for para in paras:
                m = _FOOTNOTE_LEADER_RE.match(para)
                if m:
                    footnotes[int(m.group(1))] = m.group(2).strip()
        else:
            remaining.append(el)

    return remaining, footnotes


def _attach_footnotes_to_tables(
    elements: list[ParsedElement],
    footnotes: dict[int, str],
) -> list[ParsedElement]:
    """Embed footnote texts into the metadata of financial tables that reference them.

    Looks for ``(N)`` occurrences inside the table's ``text_for_retrieval``.
    """
    if not footnotes:
        return elements

    result: list[ParsedElement] = []
    for el in elements:
        if el.element_type == "financial_table" and footnotes:
            refs = set(int(m) for m in re.findall(r"\((\d+)\)", el.text_for_retrieval))
            attached = {str(n): footnotes[n] for n in refs if n in footnotes}
            if attached:
                meta = dict(el.metadata)
                meta["footnotes"] = attached
                el = ParsedElement(
                    element_id=el.element_id,
                    element_type=el.element_type,
                    section_path=el.section_path,
                    text_for_retrieval=el.text_for_retrieval,
                    table_name=el.table_name,
                    metadata=meta,
                )
        result.append(el)
    return result


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _make_llm(temperature: float = 0.0) -> Any:
    try:
        from .llm import get_llm
    except ImportError:
        from agent.tools.llm import get_llm
    return get_llm(temperature=temperature)


def _llm_clean_narrative(llm: Any, batch: list[ParsedElement]) -> list[str]:
    """Ask LLM to remove any remaining page-header / footer lines.

    Returns one cleaned ``text_for_retrieval`` string per input element.
    """
    numbered = "\n\n".join(
        f"[{i}]\n{el.text_for_retrieval}" for i, el in enumerate(batch)
    )
    prompt = (
        "You are cleaning SEC EDGAR filing text that was extracted from HTML.\n"
        "Some paragraphs contain page-header or footer lines such as:\n"
        "  'Apple Inc. | Q3 2017 Form 10-Q | 15'\n"
        "  'PART I'\n"
        "  Company name / report type / page number combinations on their own line.\n\n"
        "For each numbered block below, return ONLY the cleaned text "
        "(remove header/footer lines, keep all substantive content). "
        "Respond with a JSON array of strings, one per block, "
        "in the same order.\n\n"
        f"{numbered}"
    )
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        results = json.loads(raw)
        if isinstance(results, list) and len(results) == len(batch):
            return [str(r) for r in results]
    except json.JSONDecodeError:
        pass
    logger.warning("[enricher] LLM clean_narrative: could not parse JSON, keeping originals")
    return [el.text_for_retrieval for el in batch]


def _llm_extract_captions(
    llm: Any,
    tables: list[ParsedElement],
    context_by_id: dict[str, str],
) -> list[str | None]:
    """Ask LLM for a concise descriptive caption for each financial table.

    ``context_by_id`` maps element_id → text of the immediately preceding narrative.
    Returns one caption (or None to keep existing) per table.
    """
    items = []
    for tbl in tables:
        ctx = context_by_id.get(tbl.element_id, "")[:300]
        # Show only first 6 rows of the markdown table to keep prompt short
        table_preview = "\n".join(tbl.text_for_retrieval.splitlines()[:8])
        items.append(
            f"element_id: {tbl.element_id}\n"
            f"current_caption: {tbl.table_name or '(none)'}\n"
            f"preceding_context: {ctx}\n"
            f"table_preview:\n{table_preview}"
        )
    combined = "\n\n---\n\n".join(items)
    prompt = (
        "You are extracting meaningful captions for financial tables in an SEC filing.\n"
        "For each table below, return a short descriptive caption (≤ 12 words) "
        "that describes WHAT the table shows (e.g. 'Condensed Consolidated Income Statement').\n"
        "If the current_caption is already good, you may keep it.\n"
        "Do NOT use page-header text like 'Apple Inc. | Form 10-Q | 7' as a caption.\n\n"
        "Respond with a JSON object: {element_id: caption_string, …}.\n\n"
        f"{combined}"
    )
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        mapping: dict[str, str] = json.loads(raw)
        return [mapping.get(tbl.element_id) for tbl in tables]
    except json.JSONDecodeError:
        logger.warning("[enricher] LLM extract_captions: could not parse JSON")
        return [None] * len(tables)


def _llm_detect_subsections(
    llm: Any,
    el: ParsedElement,
) -> list[tuple[str | None, str]]:
    """Ask LLM to identify implicit sub-section boundaries inside a long narrative.

    Returns list of (sub_heading | None, body_text) pairs.
    """
    prompt = (
        "The following is a long narrative block from an SEC 10-Q filing. "
        "Identify any implicit sub-section titles (short standalone lines or "
        "topic changes that function as headings, e.g. 'iPhone', 'Cash Flow Hedges').\n\n"
        "Split the text at those boundaries. "
        "Return a JSON array of objects: "
        '[{"heading": "<title or null>", "text": "<body text>"}, ...].\n\n'
        "Keep the text content verbatim. "
        "If there are no meaningful boundaries, return a single object with heading=null.\n\n"
        f"TEXT:\n{el.text_for_retrieval}"
    )
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        parts = json.loads(raw)
        if isinstance(parts, list) and parts:
            return [(p.get("heading"), p.get("text", "")) for p in parts]
    except json.JSONDecodeError:
        pass
    logger.warning("[enricher] LLM detect_subsections: could not parse JSON, keeping original")
    return [(None, el.text_for_retrieval)]


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------


class EdgarEnricher:
    """Applies three enrichment passes to a list of ParsedElement objects.

    Pass 1 (rules)  – strip page stamps, split obvious implicit headings,
                       extract and attach footnotes.
    Pass 2 (LLM)    – clean residual headers/footers, fix table captions,
                       detect sub-sections in long MD&A narratives.
    """

    def __init__(self, *, use_llm: bool = True) -> None:
        self._use_llm = use_llm
        self._llm: Any = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = _make_llm()
        return self._llm

    # ------------------------------------------------------------------
    # Pass 1: rules
    # ------------------------------------------------------------------

    def _pass_rules(
        self, elements: list[ParsedElement]
    ) -> list[ParsedElement]:
        # 1a. Strip page stamps from all narratives
        cleaned: list[ParsedElement] = []
        for el in elements:
            if el.element_type == "narrative":
                text, n_removed = _strip_page_stamps(el.text_for_retrieval)
                meta = dict(el.metadata)
                if n_removed:
                    meta["page_stamps_removed"] = n_removed
                cleaned.append(
                    ParsedElement(
                        element_id=el.element_id,
                        element_type=el.element_type,
                        section_path=el.section_path,
                        text_for_retrieval=text,
                        table_name=el.table_name,
                        metadata=meta,
                    )
                )
            else:
                cleaned.append(el)

        # 1b. Extract standalone footnote elements and attach to tables
        without_footnotes, footnotes = _extract_footnotes(cleaned)
        with_footnotes = _attach_footnotes_to_tables(without_footnotes, footnotes)
        logger.info(
            "[enricher] Rules pass: {} page stamps stripped, {} footnotes extracted",
            sum(e.metadata.get("page_stamps_removed", 0) for e in with_footnotes),
            len(footnotes),
        )

        # 1c. Rule-based implicit heading split (short title-case lines)
        expanded: list[ParsedElement] = []
        for el in with_footnotes:
            if el.element_type != "narrative" or len(el.text_for_retrieval) < _SUBHEAD_MIN_CHARS:
                expanded.append(el)
                continue
            sections = _split_implicit_headings(el.text_for_retrieval)
            if len(sections) <= 1:
                expanded.append(el)
                continue
            for i, (heading, body) in enumerate(sections):
                if not body.strip():
                    continue
                new_path = list(el.section_path)
                if heading:
                    new_path = list(el.section_path) + [heading]
                meta = dict(el.metadata)
                meta["split_from"] = el.element_id
                expanded.append(
                    ParsedElement(
                        element_id=f"{el.element_id}_s{i}",
                        element_type="narrative",
                        section_path=new_path,
                        text_for_retrieval=body,
                        table_name=None,
                        metadata=meta,
                    )
                )

        # 1d. Drop tiny narrative fragments (P3)
        expanded = [
            el for el in expanded
            if el.element_type != "narrative"
            or len(el.text_for_retrieval.strip()) >= _MIN_NARRATIVE_CHARS
        ]

        # 1e. Deduplicate by first-200-chars fingerprint (P4)
        seen: set[str] = set()
        deduped: list[ParsedElement] = []
        for el in expanded:
            key = el.text_for_retrieval[:200]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(el)

        # 1f. Back-fill section_path for cover / TOC elements (P5)
        # All elements that appear before any Part/Item heading (parser left
        # section_path empty) belong to the front-matter "Cover" section.
        # The TABLE OF CONTENTS table is also front-matter; giving it a
        # synthetic ["Table of Contents"] path would create a phantom section
        # in the hierarchy — keep it under ["Cover"] for consistency.
        result: list[ParsedElement] = []
        for el in deduped:
            if el.section_path:
                result.append(el)
                continue
            result.append(
                ParsedElement(
                    element_id=el.element_id,
                    element_type=el.element_type,
                    section_path=["Cover"],
                    text_for_retrieval=el.text_for_retrieval,
                    table_name=el.table_name,
                    metadata=el.metadata,
                )
            )

        # 1g. Migrate ALL-CAPS / unit-note lines from narrative into the
        #     immediately following financial table element.
        #
        # Two sub-cases:
        #   A. Narrative consists ENTIRELY of table-header-like paragraphs
        #      (every paragraph matches _looks_like_table_header).
        #      → Drop the narrative; fold the headers into the table's caption
        #        and metadata, regardless of whether the table already has a
        #        good caption.
        #   B. Only the trailing paragraphs are table-header-like.
        #      → Strip those tail paragraphs from the narrative, update the
        #        table's caption if it was bad, keep the rest of the narrative.
        #
        # In both cases the title lines are stored in
        # metadata["table_headers_from_narrative"] on the table element so
        # context is never lost.
        migrated_headers = 0
        migrated: list[ParsedElement] = []
        for i, el in enumerate(result):
            if (
                el.element_type == "narrative"
                and i + 1 < len(result)
                and result[i + 1].element_type == "financial_table"
            ):
                paras = [p for p in el.text_for_retrieval.strip().split("\n\n") if p.strip()]
                # Peel trailing header-like paragraphs from the bottom
                tail_headers: list[str] = []
                body_paras: list[str] = list(paras)
                while body_paras and _looks_like_table_header(body_paras[-1]):
                    tail_headers.insert(0, body_paras.pop().strip())

                if tail_headers:
                    next_tbl = result[i + 1]
                    tbl_meta = dict(next_tbl.metadata)
                    tbl_meta["table_headers_from_narrative"] = tail_headers

                    # Build caption: prefer the first ALL-CAPS title; fall
                    # back to first header if none found.
                    allcaps = next(
                        (h for h in tail_headers if _looks_like_table_header(h) and not h.startswith("(")),
                        tail_headers[0],
                    )
                    # Update caption only when the existing one is bad
                    if _bad_caption(next_tbl.table_name):
                        new_caption = allcaps
                        tbl_text = next_tbl.text_for_retrieval
                        if tbl_text.startswith("## "):
                            tbl_text = re.sub(r"^##[^\n]*\n\n", f"## {new_caption}\n\n", tbl_text)
                        else:
                            tbl_text = f"## {new_caption}\n\n{tbl_text}"
                        result[i + 1] = ParsedElement(
                            element_id=next_tbl.element_id,
                            element_type=next_tbl.element_type,
                            section_path=next_tbl.section_path,
                            text_for_retrieval=tbl_text,
                            table_name=new_caption,
                            metadata=tbl_meta,
                        )
                    else:
                        # Caption is already good — just attach context metadata
                        result[i + 1] = ParsedElement(
                            element_id=next_tbl.element_id,
                            element_type=next_tbl.element_type,
                            section_path=next_tbl.section_path,
                            text_for_retrieval=next_tbl.text_for_retrieval,
                            table_name=next_tbl.table_name,
                            metadata=tbl_meta,
                        )
                    migrated_headers += 1

                    remaining = "\n\n".join(body_paras).strip()
                    # Case A: nothing left → drop the narrative entirely
                    if not remaining or len(remaining) < _MIN_NARRATIVE_CHARS:
                        continue  # do not append the original element
                    # Case B: non-header body remains → keep trimmed narrative
                    meta = dict(el.metadata)
                    meta["table_header_migrated"] = tail_headers
                    migrated.append(
                        ParsedElement(
                            element_id=el.element_id,
                            element_type=el.element_type,
                            section_path=el.section_path,
                            text_for_retrieval=remaining,
                            table_name=None,
                            metadata=meta,
                        )
                    )
                    continue

            migrated.append(el)

        if migrated_headers:
            logger.info("[enricher] Rules pass: {} table headers migrated from narratives", migrated_headers)
        return migrated

    # ------------------------------------------------------------------
    # Pass 2: LLM
    # ------------------------------------------------------------------

    def _pass_llm(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        llm = self._get_llm()

        # 2a. LLM-clean narratives that still contain possible header/footer text
        #     (heuristic: contains a | pipe | in any line)
        dirty_indices = [
            i for i, el in enumerate(elements)
            if el.element_type == "narrative" and "|" in el.text_for_retrieval
        ]
        if dirty_indices:
            # Process in batches of 8
            for batch_start in range(0, len(dirty_indices), 8):
                batch_idx = dirty_indices[batch_start: batch_start + 8]
                batch = [elements[i] for i in batch_idx]
                cleaned_texts = _llm_clean_narrative(llm, batch)
                for i, idx in enumerate(batch_idx):
                    old = elements[idx]
                    meta = dict(old.metadata)
                    meta["llm_cleaned"] = True
                    elements[idx] = ParsedElement(
                        element_id=old.element_id,
                        element_type=old.element_type,
                        section_path=old.section_path,
                        text_for_retrieval=cleaned_texts[i],
                        table_name=old.table_name,
                        metadata=meta,
                    )
            logger.info("[enricher] LLM cleaned {} narrative elements", len(dirty_indices))

        # 2b. Fix table captions: replace page-header style captions
        #     Build preceding-narrative context map first
        context_by_id: dict[str, str] = {}
        last_nar = ""
        for el in elements:
            if el.element_type == "narrative":
                last_nar = el.text_for_retrieval[:400]
            elif el.element_type == "financial_table":
                context_by_id[el.element_id] = last_nar

        bad_caption_idx = [
            i for i, el in enumerate(elements)
            if el.element_type == "financial_table" and _bad_caption(el.table_name)
        ]
        for batch_start in range(0, len(bad_caption_idx), _CAPTION_BATCH):
            batch_idx = bad_caption_idx[batch_start: batch_start + _CAPTION_BATCH]
            batch = [elements[i] for i in batch_idx]
            new_captions = _llm_extract_captions(llm, batch, context_by_id)
            for i, idx in enumerate(batch_idx):
                if new_captions[i]:
                    old = elements[idx]
                    new_caption = new_captions[i]
                    new_text = re.sub(
                        r"^##[^\n]*\n\n",
                        f"## {new_caption}\n\n",
                        old.text_for_retrieval,
                    )
                    elements[idx] = ParsedElement(
                        element_id=old.element_id,
                        element_type=old.element_type,
                        section_path=old.section_path,
                        text_for_retrieval=new_text,
                        table_name=new_caption,
                        metadata=old.metadata,
                    )
        logger.info("[enricher] LLM fixed captions for {} tables", len(bad_caption_idx))

        # 2c. LLM sub-section detection for long MD&A narratives
        #     (rule split in pass 1 handled simple cases; LLM handles complex ones)
        mda_long = [
            i for i, el in enumerate(elements)
            if el.element_type == "narrative"
            and len(el.text_for_retrieval) > _SUBHEAD_MIN_CHARS * 2
            and any("Item 2" in s or "mda" in s.lower() for s in
                    el.section_path + [el.metadata.get("finance_section", "")])
        ]
        new_elements: list[ParsedElement] = list(elements)
        offset = 0
        for idx in mda_long:
            el = new_elements[idx + offset]
            sections = _llm_detect_subsections(llm, el)
            if len(sections) <= 1:
                continue
            replacements: list[ParsedElement] = []
            for j, (heading, body) in enumerate(sections):
                if not body.strip():
                    continue
                new_path = list(el.section_path)
                if heading:
                    new_path = list(el.section_path) + [heading]
                meta = dict(el.metadata)
                meta["llm_split_from"] = el.element_id
                replacements.append(
                    ParsedElement(
                        element_id=f"{el.element_id}_l{j}",
                        element_type="narrative",
                        section_path=new_path,
                        text_for_retrieval=body,
                        table_name=None,
                        metadata=meta,
                    )
                )
            new_elements[idx + offset: idx + offset + 1] = replacements
            offset += len(replacements) - 1
        if offset:
            logger.info("[enricher] LLM sub-section split added {} extra elements", offset)
        return new_elements

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def enrich(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        elements = self._pass_rules(elements)
        if self._use_llm:
            elements = self._pass_llm(elements)
        return elements


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def enrich_parsed_json(
    parsed_json_path: Path,
    *,
    output_path: Path | None = None,
    use_llm: bool = True,
) -> tuple[list[ParsedElement], Path]:
    data = json.loads(parsed_json_path.read_text(encoding="utf-8"))
    elements = [_dict_to_element(d) for d in data.get("elements", [])]

    enricher = EdgarEnricher(use_llm=use_llm)
    enriched = enricher.enrich(elements)

    out = output_path or parsed_json_path.with_name(
        parsed_json_path.stem + ENRICHED_JSON_SUFFIX
    )
    payload = {
        "pipeline_version": data.get("pipeline_version", "") + "+enriched",
        "source_parsed_json": str(parsed_json_path.resolve()),
        "element_count": len(enriched),
        "elements": [asdict(e) for e in enriched],
    }
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("[enricher] {} → {} elements → {}", len(elements), len(enriched), out.name)
    return enriched, out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Enrich edgar_htm_parser output: clean headers, fix captions, split sub-sections."
    )
    p.add_argument("parsed_json", type=Path, help="Path to *.parsed.json")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path")
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Run only rule-based passes (no LLM calls)",
    )
    p.add_argument("--summary", action="store_true", help="Print a short summary after enrichment")
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = _cli().parse_args(argv)
    if not args.parsed_json.is_file():
        print(f"Error: not found: {args.parsed_json}", file=sys.stderr)
        return 1

    enriched, out_path = enrich_parsed_json(
        args.parsed_json,
        output_path=args.output,
        use_llm=not args.no_llm,
    )
    print(f"Enriched {len(enriched)} elements → {out_path.resolve()}")

    if args.summary:
        from collections import Counter
        types = Counter(e.element_type for e in enriched)
        llm_cleaned = sum(1 for e in enriched if e.metadata.get("llm_cleaned"))
        footnote_tables = sum(1 for e in enriched if e.metadata.get("footnotes"))
        stamps = sum(e.metadata.get("page_stamps_removed", 0) for e in enriched)
        split = sum(1 for e in enriched if e.metadata.get("split_from") or e.metadata.get("llm_split_from"))
        print(f"  element types      : {dict(types)}")
        print(f"  page stamps removed: {stamps}")
        print(f"  LLM narrative clean: {llm_cleaned}")
        print(f"  tables with footnotes attached: {footnote_tables}")
        print(f"  sub-section splits : {split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
