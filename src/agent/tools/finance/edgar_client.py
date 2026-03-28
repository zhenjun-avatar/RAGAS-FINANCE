"""Fetch SEC EDGAR filing HTML using CIK + accession (submissions API + optional index.json)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from core.config import config

_DATA_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik10}.json"
_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{name}"


def cik_to_10(cik: int | str) -> str:
    n = int(str(cik).lstrip("0") or "0")
    return f"{n:010d}"


def accession_to_folder(accession: str) -> str:
    return re.sub(r"[^0-9]", "", accession)


def sec_http_client() -> httpx.AsyncClient:
    ua = config.sec_http_user_agent.strip()
    if "replace-with-your-email" in ua.lower() or "example.com" in ua.lower():
        # Still works for many endpoints; SEC asks for real contact in production.
        pass
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=20.0),
        headers={"User-Agent": ua, "Accept": "application/json,text/html,*/*;q=0.8"},
        follow_redirects=True,
    )


def index_recent_filings(submissions: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Map accessionNumber -> form, filingDate, primaryDocument."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    keys = ("accessionNumber", "filingDate", "form", "primaryDocument")
    arrays = {k: recent.get(k) or [] for k in keys}
    n = len(arrays["accessionNumber"])
    out: dict[str, dict[str, str]] = {}
    for i in range(n):
        acc = str(arrays["accessionNumber"][i]).strip()
        if not acc:
            continue
        out[acc] = {
            "filingDate": str(arrays["filingDate"][i]).strip() if i < len(arrays["filingDate"]) else "",
            "form": str(arrays["form"][i]).strip() if i < len(arrays["form"]) else "",
            "primaryDocument": str(arrays["primaryDocument"][i]).strip()
            if i < len(arrays["primaryDocument"])
            else "",
        }
    return out


def _pick_largest_body_htm(items: list[dict[str, Any]]) -> str | None:
    best_name: str | None = None
    best_size = -1
    for it in items:
        name = str(it.get("name") or "").strip()
        low = name.lower()
        if not low.endswith((".htm", ".html")):
            continue
        if "index.htm" in low or "index.html" in low:
            continue
        raw_sz = it.get("size")
        try:
            sz = int(str(raw_sz).strip()) if str(raw_sz).strip().isdigit() else 0
        except (TypeError, ValueError):
            sz = 0
        if sz > best_size:
            best_size = sz
            best_name = name
    return best_name


async def fetch_submissions(client: httpx.AsyncClient, cik: int | str) -> dict[str, Any]:
    url = _DATA_SUBMISSIONS.format(cik10=cik_to_10(cik))
    r = await client.get(url)
    r.raise_for_status()
    return r.json()


async def fetch_index_json(client: httpx.AsyncClient, cik: int, accession: str) -> dict[str, Any] | None:
    folder = accession_to_folder(accession)
    url = _ARCHIVES.format(cik=int(cik), folder=folder, name="index.json")
    r = await client.get(url)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


async def resolve_primary_document(
    client: httpx.AsyncClient,
    cik: int | str,
    accession: str,
    submissions_cache: dict[str, Any] | None = None,
) -> str | None:
    data = submissions_cache if submissions_cache is not None else await fetch_submissions(client, cik)
    row = index_recent_filings(data).get(accession)
    if row and row.get("primaryDocument"):
        return row["primaryDocument"]
    idx = await fetch_index_json(client, int(cik), accession)
    if not idx:
        return None
    items = ((idx.get("directory") or {}).get("item")) or []
    if not isinstance(items, list):
        return None
    return _pick_largest_body_htm(items)


@dataclass(frozen=True)
class FilingRef:
    cik: int
    accession: str
    form: str
    filing_date: str
    primary_document: str
    source_url: str


async def build_filing_ref(
    client: httpx.AsyncClient,
    cik: int | str,
    accession: str,
    submissions_cache: dict[str, Any] | None = None,
) -> FilingRef | None:
    cik_int = int(str(cik).lstrip("0") or "0")
    data = submissions_cache if submissions_cache is not None else await fetch_submissions(client, cik)
    row = index_recent_filings(data).get(accession) or {}
    primary = (row.get("primaryDocument") or "").strip() or await resolve_primary_document(
        client, cik_int, accession, submissions_cache=data
    )
    if not primary:
        return None
    folder = accession_to_folder(accession)
    url = _ARCHIVES.format(cik=cik_int, folder=folder, name=primary)
    return FilingRef(
        cik=cik_int,
        accession=accession,
        form=(row.get("form") or "").strip(),
        filing_date=(row.get("filingDate") or "").strip(),
        primary_document=primary,
        source_url=url,
    )


async def download_filing_html(client: httpx.AsyncClient, ref: FilingRef) -> bytes:
    r = await client.get(ref.source_url)
    r.raise_for_status()
    return r.content
