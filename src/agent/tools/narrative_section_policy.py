"""Scoring configuration for narrative retrieval.

Replaces the old role-based section/leaf filtering with a simple config container.
Quality gating is done via rerank_score; no section_role or leaf_role is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NarrativeSectionPolicy:
    penalize_forward_looking: bool = True
    penalize_early_position: bool = True
    pass_score_threshold: float = 2.0


def resolve_section_policy(
    narrative_targets: tuple[str, ...] | list[str] = (),
) -> NarrativeSectionPolicy:
    targets = frozenset(str(t).strip().lower() for t in narrative_targets if str(t).strip())
    # MDA-like questions benefit from stricter forward-looking / early-position penalties
    is_mda_like = bool(targets & {"management_discussion", "margin_cost_structure", "liquidity"})
    return NarrativeSectionPolicy(
        penalize_forward_looking=is_mda_like,
        penalize_early_position=is_mda_like,
        pass_score_threshold=2.0 if is_mda_like else 1.5,
    )
