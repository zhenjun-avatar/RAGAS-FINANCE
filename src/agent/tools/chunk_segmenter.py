"""Structure-aware chunking utilities for node-centric ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, List, Optional


CHUNK_SIZE = 1500
OVERLAP_SIZE = 220
MIN_CHUNK_SIZE = 300

_PAGE_MARKER_RE = re.compile(r"^##\s*第\s*(\d+)\s*页\s*$")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千0-9]+[章节部分篇卷]|[0-9]+(?:\.[0-9]+){0,3}|[一二三四五六七八九十]+、|[（(][0-9一二三四五六七八九十]+[)）])\s*(.+)?$"
)
_SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？!?；;：:\.])\s+")


@dataclass
class _Block:
    text: str
    block_type: str
    page_number: Optional[int] = None
    heading_path: list[str] = field(default_factory=list)
    heading_level: Optional[int] = None


@dataclass
class ChunkPayload:
    text: str
    title: Optional[str]
    metadata: dict[str, Any]


def segment_by_fixed_size(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> List[str]:
    return [chunk.text for chunk in build_retrieval_chunks(text, chunk_size, overlap, min_chunk_size)]


def segment_by_sentences(
    text: str,
    target_chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
) -> List[str]:
    return [chunk.text for chunk in build_retrieval_chunks(text, target_chunk_size, overlap, MIN_CHUNK_SIZE)]


def build_retrieval_chunks(
    text: str,
    target_chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[ChunkPayload]:
    if not text or not text.strip():
        return []

    blocks = _parse_blocks(text, target_chunk_size)
    if not blocks:
        return []

    chunks: list[ChunkPayload] = []
    current: list[_Block] = []

    def flush(*, carry_overlap: bool) -> None:
        nonlocal current
        if not current:
            return
        chunks.append(_build_chunk(current))
        current = _select_overlap_blocks(current, overlap) if carry_overlap else []

    for block in blocks:
        boundary_block = block.block_type in {"heading", "page_marker"}
        current_chars = _blocks_length(current)
        next_chars = current_chars + len(block.text) + (2 if current else 0)

        if boundary_block and current and any(item.block_type == "paragraph" for item in current):
            flush(carry_overlap=False)

        if current and next_chars > target_chunk_size and current_chars >= min_chunk_size:
            flush(carry_overlap=not boundary_block)

        current.append(block)

    if current:
        flush(carry_overlap=False)

    return _merge_small_tail_chunks(chunks, min_chunk_size)


def _parse_blocks(text: str, target_chunk_size: int) -> list[_Block]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]

    blocks: list[_Block] = []
    current_page: Optional[int] = None
    heading_path: list[str] = []

    for paragraph in paragraphs:
        page_match = _PAGE_MARKER_RE.match(paragraph)
        if page_match:
            current_page = int(page_match.group(1))
            blocks.append(
                _Block(
                    text=paragraph,
                    block_type="page_marker",
                    page_number=current_page,
                    heading_path=list(heading_path),
                )
            )
            continue

        heading = _detect_heading(paragraph)
        if heading:
            level, title = heading
            heading_path = _update_heading_path(heading_path, level, title)
            blocks.append(
                _Block(
                    text=title,
                    block_type="heading",
                    page_number=current_page,
                    heading_path=list(heading_path),
                    heading_level=level,
                )
            )
            continue

        for part in _split_long_paragraph(paragraph, target_chunk_size):
            clean = part.strip()
            if not clean:
                continue
            blocks.append(
                _Block(
                    text=clean,
                    block_type="paragraph",
                    page_number=current_page,
                    heading_path=list(heading_path),
                )
            )

    return blocks


def _detect_heading(paragraph: str) -> tuple[int, str] | None:
    markdown_match = _MARKDOWN_HEADING_RE.match(paragraph)
    if markdown_match:
        return len(markdown_match.group(1)), markdown_match.group(2).strip()

    numbered_match = _NUMBERED_HEADING_RE.match(paragraph)
    if numbered_match and len(paragraph) <= 120:
        suffix = (numbered_match.group(1) or "").strip()
        parts = re.findall(r"\d+", paragraph)
        level = min(4, len(parts)) if parts else 1
        title = paragraph.strip() if not suffix else paragraph.strip()
        return level, title

    return None


def _update_heading_path(current: list[str], level: int, title: str) -> list[str]:
    trimmed = current[: max(0, level - 1)]
    trimmed.append(title)
    return trimmed


def _split_long_paragraph(paragraph: str, target_chunk_size: int) -> list[str]:
    if len(paragraph) <= target_chunk_size:
        return [paragraph]

    sentences = [part.strip() for part in _SENTENCE_BREAK_RE.split(paragraph) if part.strip()]
    if len(sentences) <= 1:
        return [paragraph[i : i + target_chunk_size] for i in range(0, len(paragraph), target_chunk_size)]

    groups: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= target_chunk_size:
            current = candidate
            continue
        if current:
            groups.append(current)
            current = sentence
        else:
            groups.extend(
                sentence[i : i + target_chunk_size].strip()
                for i in range(0, len(sentence), target_chunk_size)
                if sentence[i : i + target_chunk_size].strip()
            )
            current = ""
    if current:
        groups.append(current)
    return groups


def _select_overlap_blocks(blocks: list[_Block], overlap_chars: int) -> list[_Block]:
    carried: list[_Block] = []
    total = 0
    for block in reversed(blocks):
        if block.block_type in {"heading", "page_marker"}:
            break
        block_len = len(block.text)
        if carried and total + block_len > overlap_chars:
            break
        carried.insert(0, block)
        total += block_len
    return carried


def _blocks_length(blocks: list[_Block]) -> int:
    if not blocks:
        return 0
    return sum(len(block.text) for block in blocks) + (len(blocks) - 1) * 2


def _build_chunk(blocks: list[_Block]) -> ChunkPayload:
    texts = [block.text for block in blocks if block.text.strip()]
    pages = [block.page_number for block in blocks if block.page_number is not None]
    heading_paths = [block.heading_path for block in blocks if block.heading_path]
    active_heading_path = heading_paths[-1] if heading_paths else []
    section_title = active_heading_path[-1] if active_heading_path else None
    title = section_title or (f"Page {pages[0]}" if pages else None)
    metadata: dict[str, Any] = {
        "block_count": len(blocks),
        "char_count": len("\n\n".join(texts)),
    }
    if pages:
        metadata["page_start"] = min(pages)
        metadata["page_end"] = max(pages)
    if active_heading_path:
        metadata["heading_path"] = active_heading_path
        metadata["section_title"] = section_title
        metadata["section_depth"] = len(active_heading_path)
    return ChunkPayload(text="\n\n".join(texts).strip(), title=title, metadata=metadata)


def _merge_small_tail_chunks(chunks: list[ChunkPayload], min_chunk_size: int) -> list[ChunkPayload]:
    if len(chunks) < 2:
        return [chunk for chunk in chunks if chunk.text.strip()]

    merged: list[ChunkPayload] = []
    for chunk in chunks:
        same_section = bool(
            merged
            and merged[-1].metadata.get("section_title") == chunk.metadata.get("section_title")
        )
        has_no_section = not merged or (
            merged[-1].metadata.get("section_title") is None
            and chunk.metadata.get("section_title") is None
        )
        if merged and len(chunk.text) < min_chunk_size and (same_section or has_no_section):
            prev = merged.pop()
            merged_text = f"{prev.text}\n\n{chunk.text}".strip()
            merged_metadata = dict(prev.metadata)
            merged_metadata["block_count"] = int(prev.metadata.get("block_count", 0)) + int(
                chunk.metadata.get("block_count", 0)
            )
            merged_metadata["char_count"] = len(merged_text)
            if "page_start" in prev.metadata or "page_start" in chunk.metadata:
                merged_metadata["page_start"] = min(
                    int(prev.metadata.get("page_start") or chunk.metadata.get("page_start") or 0),
                    int(chunk.metadata.get("page_start") or prev.metadata.get("page_start") or 0),
                )
                merged_metadata["page_end"] = max(
                    int(prev.metadata.get("page_end") or chunk.metadata.get("page_end") or 0),
                    int(chunk.metadata.get("page_end") or prev.metadata.get("page_end") or 0),
                )
            merged.append(
                ChunkPayload(
                    text=merged_text,
                    title=prev.title or chunk.title,
                    metadata=merged_metadata,
                )
            )
        else:
            merged.append(chunk)
    return [chunk for chunk in merged if chunk.text.strip()]
