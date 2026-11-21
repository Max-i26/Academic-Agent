"""Reciprocal Rank Fusion (RRF) implementation for hybrid retrieval."""

import logging
from typing import Any
from academic_agent.retrieval.models import RetrievalQuery, RetrievalCandidate

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """Combines ranked candidate lists across multiple retrieval methods using RRF.

    Formula:
        RRF(d) = sum_{m in M} ( 1 / (k + rank_m(d)) )

    Guarantees:
    - Merge identity strictly based on record_id (fragment-aware).
    - Preservation of original branch scores and method attribution.
    - Fully deterministic 4-tier tie breaking.
    - Protection against duplicate candidates within a single method.
    """

    def __init__(self, k: int = 60):
        """Initialize RRF with smoothing constant k.

        Args:
            k: Smoothing constant balancing high-ranked items vs consensus (default: 60).
        """
        if k < 1:
            raise ValueError(f"RRF smoothing constant k must be >= 1, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        """Name of the fusion algorithm."""
        return "rrf"

    @property
    def k(self) -> int:
        """The smoothing constant."""
        return self._k

    def fuse(
        self,
        query: RetrievalQuery,
        candidates_by_method: dict[str, list[RetrievalCandidate]],
        top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Merge candidate lists from different retrieval methods.

        Args:
            query: The originating RetrievalQuery.
            candidates_by_method: Mapping from method name ('dense', 'sparse', etc.) to candidates.
            top_k: Number of fused candidates to return. If None, uses query.top_k.

        Returns:
            Unified, deduplicated, and ranked list of RetrievalCandidates.
        """
        target_k = top_k if top_k is not None else query.top_k
        if target_k <= 0:
            return []

        if not candidates_by_method:
            return []

        # Data structure for accumulating candidates by record_id
        # record_id -> dict with aggregated metadata and score
        fused_entries: dict[str, dict[str, Any]] = {}

        for method_name, candidate_list in candidates_by_method.items():
            if not candidate_list:
                continue

            seen_in_branch: set[str] = set()

            for cand in candidate_list:
                rid = cand.record_id
                if not rid:
                    continue

                # Defensive check against duplicate record_ids within the same branch
                if rid in seen_in_branch:
                    logger.warning(
                        "Duplicate record_id '%s' detected within method '%s'. Ignoring duplicate.",
                        rid,
                        method_name,
                    )
                    continue
                seen_in_branch.add(rid)

                if cand.rank < 1:
                    raise ValueError(f"Candidate rank must be >= 1, got {cand.rank} for '{rid}'")

                # Compute this method's reciprocal rank contribution
                rr_contribution = 1.0 / (self._k + cand.rank)

                if rid not in fused_entries:
                    fused_entries[rid] = {
                        "record_id": rid,
                        "chunk_id": cand.chunk_id,
                        "text": cand.text,
                        "source_file": cand.source_file,
                        "source_pages": list(cand.source_pages),
                        "source_unit_ids": list(cand.source_unit_ids),
                        "section_title": cand.section_title,
                        "content_type": cand.content_type,
                        "fragment_index": cand.fragment_index,
                        "total_fragments": cand.total_fragments,
                        "rrf_score": rr_contribution,
                        "contributing_methods": [method_name],
                        "method_scores": {method_name: cand.score},
                        "method_ranks": {method_name: cand.rank},
                        "metadata": dict(cand.metadata),
                    }
                else:
                    entry = fused_entries[rid]
                    entry["rrf_score"] += rr_contribution
                    entry["contributing_methods"].append(method_name)
                    entry["method_scores"][method_name] = cand.score
                    entry["method_ranks"][method_name] = cand.rank
                    entry["metadata"].update(cand.metadata)

        if not fused_entries:
            return []

        # Deterministic 4-tier tie breaking:
        # 1. RRF score descending (-score)
        # 2. Method consensus count descending (-len(methods))
        # 3. Best individual rank ascending (min(ranks))
        # 4. Lexicographical record_id ascending
        sorted_entries = sorted(
            fused_entries.values(),
            key=lambda e: (
                -e["rrf_score"],
                -len(e["contributing_methods"]),
                min(e["method_ranks"].values()),
                e["record_id"],
            ),
        )

        fused_candidates: list[RetrievalCandidate] = []
        for rank_idx, entry in enumerate(sorted_entries[:target_k], start=1):
            fused_cand = RetrievalCandidate(
                record_id=entry["record_id"],
                chunk_id=entry["chunk_id"],
                rank=rank_idx,
                retrieval_method="hybrid",
                score=entry["rrf_score"],
                normalized_score=None,
                text=entry["text"],
                source_file=entry["source_file"],
                source_pages=entry["source_pages"],
                source_unit_ids=entry["source_unit_ids"],
                section_title=entry["section_title"],
                content_type=entry["content_type"],
                fragment_index=entry["fragment_index"],
                total_fragments=entry["total_fragments"],
                contributing_methods=entry["contributing_methods"],
                method_scores=entry["method_scores"],
                method_ranks=entry["method_ranks"],
                metadata={
                    **entry["metadata"],
                    "rrf_score": entry["rrf_score"],
                    "agreement_count": len(entry["contributing_methods"]),
                },
            )
            fused_candidates.append(fused_cand)

        return fused_candidates
