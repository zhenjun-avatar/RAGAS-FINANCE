"""Async RAGAS evaluation worker and Langfuse score writeback."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from core.config import config
from .langfuse_tracing import tracer
from .llm import get_llm
from .node_repository import complete_evaluation_job, list_pending_evaluation_jobs

try:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
except ImportError:  # pragma: no cover
    SingleTurnSample = None
    Faithfulness = None
    LLMContextPrecisionWithoutReference = None


def _retrieved_context_texts(raw: Any) -> list[str]:
    """Normalize DB/enqueue shapes: JSONB may come back as str; items may be dict or plain str."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(str(item.get("text", "") or ""))
        elif isinstance(item, str):
            out.append(item)
        else:
            out.append(str(item))
    return [t for t in out if t.strip()]


async def _score_job(job: dict[str, Any]) -> dict[str, float]:
    """Run RAGAS metrics (caller must ensure ragas is enabled and imports succeeded)."""
    if SingleTurnSample is None or Faithfulness is None or LLMContextPrecisionWithoutReference is None:
        raise RuntimeError("RAGAS metric classes not imported")
    llm = get_llm(
        model_name=config.ragas_llm_model,
        temperature=0.0,
        ragas_strip_json_fence=True,
    )
    sample = SingleTurnSample(
        user_input=job["query"],
        response=job["answer"],
        retrieved_contexts=_retrieved_context_texts(job.get("context_json")),
    )
    faithfulness = Faithfulness(llm=llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=llm)
    return {
        "faithfulness": float(await faithfulness.single_turn_ascore(sample)),
        "context_precision": float(await context_precision.single_turn_ascore(sample)),
    }


async def run_pending_evaluations(limit: int | None = None) -> dict[str, Any]:
    jobs = await list_pending_evaluation_jobs(limit or config.ragas_batch_size)
    processed = 0
    failed = 0
    skipped = 0
    for job in jobs:
        try:
            if not config.ragas_enabled:
                await complete_evaluation_job(
                    job["id"],
                    skipped_reason="RAGAS_ENABLED=false; set RAGAS_ENABLED=true in .env, restart API, then enqueue new jobs",
                )
                skipped += 1
                continue
            if (
                SingleTurnSample is None
                or Faithfulness is None
                or LLMContextPrecisionWithoutReference is None
            ):
                await complete_evaluation_job(
                    job["id"],
                    skipped_reason="ragas import failed; install ragas (see pyproject.toml)",
                )
                skipped += 1
                continue
            scores = await _score_job(job)
            for name, value in scores.items():
                tracer.score_trace(job["trace_id"], name=name, value=value)
            await complete_evaluation_job(job["id"])
            processed += 1
        except Exception as exc:
            logger.exception("[RAGAS] Failed to evaluate job %s", job["id"])
            await complete_evaluation_job(job["id"], error=str(exc))
            failed += 1
    if skipped and not processed and not failed:
        logger.warning(
            "[RAGAS] {} job(s) marked skipped (no LLM/RAGAS run). {}",
            skipped,
            "Enable RAGAS_ENABLED or fix ragas install.",
        )
    return {"processed": processed, "failed": failed, "skipped": skipped}
