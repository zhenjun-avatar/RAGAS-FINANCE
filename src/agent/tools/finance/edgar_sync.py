"""Download EDGAR filings referenced by a companyfacts JSON and ingest as separate documents."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from tools.finance.edgar_client import build_filing_ref, download_filing_html, fetch_submissions, sec_http_client
from tools.finance.sec_company_facts import is_sec_company_facts_payload, list_accessions_from_company_facts
from tools.ingestion_service import process_edgar_filing_document


async def sync_filings_from_company_facts_file(
    *,
    company_facts_path: str | Path,
    document_id_start: int,
    max_filings: int = 30,
    accessions: set[str] | None = None,
    download_dir: str | Path | None = None,
    ingest_documents: bool = True,
    delay_seconds: float = 0.12,
) -> dict[str, Any]:
    """Each accession -> one new document_id (start, start+1, ...)."""
    path = Path(company_facts_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not is_sec_company_facts_payload(data):
        return {"success": False, "error": "Not a SEC companyfacts JSON (need cik + facts)"}
    cik = data.get("cik")
    entity = data.get("entityName")
    ordered = list_accessions_from_company_facts(data)
    if accessions is not None:
        ordered = [a for a in ordered if a in accessions]
    ordered = ordered[: max(0, max_filings)]

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    saved_files: list[str] = []
    target_dir: Path | None = None
    if download_dir:
        target_dir = Path(download_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    async with sec_http_client() as client:
        try:
            submissions = await fetch_submissions(client, cik)
        except Exception as exc:
            return {"success": False, "error": f"submissions fetch failed: {exc}"}

        doc_id = document_id_start
        for acc in ordered:
            try:
                ref = await build_filing_ref(client, cik, acc, submissions_cache=submissions)
                if not ref:
                    errors.append({"accession": acc, "error": "could not resolve primary document"})
                    continue
                raw = await download_filing_html(client, ref)
                suffix = Path(ref.primary_document).suffix or ".htm"
                persisted_path: str | None = None
                if target_dir is not None:
                    safe_name = f"EDGAR_{int(cik)}_{acc}{suffix}"
                    out_path = target_dir / safe_name
                    out_path.write_bytes(raw)
                    persisted_path = str(out_path.resolve())
                    saved_files.append(persisted_path)

                tmp_path: str | None = None
                try:
                    if ingest_documents:
                        if persisted_path:
                            ingest_path = persisted_path
                        else:
                            fd, tmp_path = tempfile.mkstemp(prefix="edgar_", suffix=suffix)
                            os.close(fd)
                            Path(tmp_path).write_bytes(raw)
                            ingest_path = tmp_path
                        out = await process_edgar_filing_document(
                            ingest_path,
                            doc_id,
                            cik=int(cik),
                            accession=ref.accession,
                            form=ref.form,
                            filed=ref.filing_date or None,
                            entity_name=str(entity) if entity else None,
                            source_url=ref.source_url,
                            primary_document=ref.primary_document,
                        )
                        if persisted_path:
                            out["downloaded_file"] = persisted_path
                        results.append({"accession": acc, "document_id": doc_id, **out})
                        doc_id += 1
                    else:
                        results.append(
                            {
                                "accession": acc,
                                "document_id": None,
                                "success": True,
                                "downloaded_file": persisted_path,
                                "source_url": ref.source_url,
                                "primary_document": ref.primary_document,
                                "ingestion_mode": "download_only",
                            }
                        )
                finally:
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
            except Exception as exc:
                logger.warning("[edgar_sync] accession {}: {}", acc, exc)
                errors.append({"accession": acc, "error": str(exc)})
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    return {
        "success": True,
        "cik": cik,
        "entity_name": entity,
        "ingested": results,
        "errors": errors,
        "ingest_documents": ingest_documents,
        "saved_file_count": len(saved_files),
        "saved_files": saved_files,
        "next_document_id": document_id_start + len(results),
    }
