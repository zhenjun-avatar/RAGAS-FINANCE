"""Async RAGAS evaluation worker and Langfuse score writeback."""

from __future__ import annotations

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


async def _score_job(job: dict[str, Any]) -> dict[str, float]:
    if not config.ragas_enabled or SingleTurnSample is None:
        return {}
    llm = get_llm(model_name=config.ragas_llm_model, temperature=0.0)
    sample = SingleTurnSample(
        user_input=job["query"],
        response=job["answer"],
        retrieved_contexts=[item.get("text", "") for item in job["context_json"]],
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
    for job in jobs:
        try:
            scores = await _score_job(job)
            for name, value in scores.items():
                tracer.score_trace(job["trace_id"], name=name, value=value)
            await complete_evaluation_job(job["id"])
            processed += 1
        except Exception as exc:
            logger.exception("[RAGAS] Failed to evaluate job %s", job["id"])
            await complete_evaluation_job(job["id"], error=str(exc))
            failed += 1
    return {"processed": processed, "failed": failed}
