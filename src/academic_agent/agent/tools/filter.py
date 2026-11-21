"""Evidence filtering tool with strict omission auditing and provenance preservation."""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from academic_agent.agent.models import ToolSideEffectPolicy
from academic_agent.agent.tools.base import BaseAgentTool


class EvidenceFilterArgs(BaseModel):
    """Input parameters for evidence filtering and prioritization."""

    candidates: List[Dict[str, Any]] = Field(
        min_length=0,
        description="List of candidate evidence dictionaries surfaced by academic_search.",
    )
    min_score: Optional[float] = Field(
        default=None,
        description="Minimum relevance score threshold. Candidates below this are audited as omitted.",
    )
    max_candidates: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of highest-scoring candidates to retain.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum cumulative token budget for retained candidates.",
    )
    required_source_file: Optional[str] = Field(
        default=None,
        description="Filter requiring candidate source_file to match this lecture PDF name.",
    )
    required_course_id: Optional[str] = Field(
        default=None,
        description="Filter requiring candidate metadata course_id to match this course identifier.",
    )


class EvidenceFilterTool(BaseAgentTool):
    """Filters candidate evidence passages while rigorously recording omission rationale."""

    name = "evidence_filter"
    description = (
        "Filter and prioritize retrieved academic evidence candidates based on relevance score "
        "thresholds, lecture boundaries, and token budgets while auditing every excluded candidate."
    )
    side_effect_policy = ToolSideEffectPolicy(
        read_only=True,
        idempotent=True,
        mutating=False,
        requires_confirmation=False,
        timeout_seconds=5.0,
    )
    args_model = EvidenceFilterArgs

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for a text passage conservatively (approx 1.3 tokens/word)."""
        if not text:
            return 0
        words = len(text.strip().split())
        return max(1, math.ceil(words * 1.3))

    def _run(self, args: EvidenceFilterArgs) -> Dict[str, Any]:
        """Filter evidence candidates while strictly preserving omission audit trail."""
        # Sort candidates descending by score if available
        sorted_candidates = sorted(
            args.candidates,
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )

        retained: List[Dict[str, Any]] = []
        omitted: List[Dict[str, Any]] = []
        cumulative_tokens: int = 0

        for cand in sorted_candidates:
            cid = cand.get("record_id") or cand.get("chunk_id") or "unknown_id"
            source_file = cand.get("source_file") or ""
            score = float(cand.get("score", 0.0))
            meta = cand.get("metadata") or {}
            course_id = meta.get("course_id") or cand.get("course_id")

            # 1. Source file constraint check
            if args.required_source_file and source_file.lower().strip() != args.required_source_file.lower().strip():
                omitted.append({
                    "record_id": cid,
                    "score": score,
                    "source_file": source_file,
                    "omission_reason": f"Source lecture file '{source_file}' does not match required '{args.required_source_file}'",
                    "omission_category": "source_mismatch",
                })
                continue

            # 2. Course ID constraint check
            if args.required_course_id and course_id and course_id.lower().strip() != args.required_course_id.lower().strip():
                omitted.append({
                    "record_id": cid,
                    "score": score,
                    "source_file": source_file,
                    "omission_reason": f"Course ID '{course_id}' does not match required '{args.required_course_id}'",
                    "omission_category": "course_mismatch",
                })
                continue

            # 3. Minimum score threshold check
            if args.min_score is not None and score < args.min_score:
                omitted.append({
                    "record_id": cid,
                    "score": score,
                    "source_file": source_file,
                    "omission_reason": f"Relevance score {score:.4f} is below minimum threshold {args.min_score:.4f}",
                    "omission_category": "below_score_threshold",
                })
                continue

            # 4. Maximum candidates ceiling check
            if args.max_candidates is not None and len(retained) >= args.max_candidates:
                omitted.append({
                    "record_id": cid,
                    "score": score,
                    "source_file": source_file,
                    "omission_reason": f"Max candidates ceiling reached ({len(retained)}/{args.max_candidates})",
                    "omission_category": "max_candidates_exceeded",
                })
                continue

            # 5. Token budget ceiling check
            cand_tokens = self._estimate_tokens(cand.get("text", ""))
            if args.max_tokens is not None and (cumulative_tokens + cand_tokens) > args.max_tokens:
                omitted.append({
                    "record_id": cid,
                    "score": score,
                    "source_file": source_file,
                    "omission_reason": f"Token budget exceeded: candidate tokens ({cand_tokens}) would push total ({cumulative_tokens + cand_tokens}) past limit ({args.max_tokens})",
                    "omission_category": "token_budget_exceeded",
                })
                continue

            # Qualified for retention
            retained.append(cand)
            cumulative_tokens += cand_tokens

        return {
            "retained_candidates": retained,
            "omitted_candidates": omitted,
            "summary": {
                "total_evaluated": len(sorted_candidates),
                "retained_count": len(retained),
                "omitted_count": len(omitted),
                "cumulative_tokens_estimated": cumulative_tokens,
            },
        }
