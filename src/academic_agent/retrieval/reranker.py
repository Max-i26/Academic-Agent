"""Cross-Encoder Reranker implementing RerankerProtocol for semantic relevance refinement."""

import logging
import time
from typing import Any, Sequence

from academic_agent.retrieval.models import RetrievalQuery, RetrievalCandidate
from academic_agent.retrieval.protocols import RerankerProtocol

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Post-retrieval candidate reranker powered by a pretrained Cross-Encoder.

    Satisfies RerankerProtocol:
    - Performs joint self-attention over (query, passage) pairs to score deep relevance.
    - Prevents single-branch dilution by evaluating semantic match across dense and sparse candidates.
    - Preserves all original candidate provenance, fragment identifiers, and historical method scores.
    - Applies deterministic 3-tier tie breaking: rerank_score -> previous_score -> record_id.
    - Employs lazy model loading with configurable device (CPU/CUDA) and batching.
    - Actively audits and diagnoses potential input passage truncation against the model token budget.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
        lazy: bool = True,
    ):
        """Initialize CrossEncoderReranker.

        Args:
            model_name: HuggingFace model hub path or local directory.
            device: Compute device ('cpu', 'cuda', etc.). Defaults to 'cpu'.
            max_length: Maximum combined token sequence length (query + candidate passage).
            batch_size: Batch size for cross-encoder inference.
            lazy: If True, delays loading model weights until the first rerank call.
        """
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._model: Any = None
        self._truncation_count: int = 0

        if not lazy:
            self.load()

    @property
    def name(self) -> str:
        """Name of the underlying reranker model."""
        return self._model_name

    @property
    def device(self) -> str:
        """Compute device used by the reranker."""
        return self._device

    @property
    def max_length(self) -> int:
        """Maximum token length for joint query-passage encoding."""
        return self._max_length

    @property
    def truncation_count(self) -> int:
        """Cumulative count of candidate passages that triggered truncation diagnostics."""
        return self._truncation_count

    def load(self) -> None:
        """Load the CrossEncoder model weights into memory if not already loaded."""
        if self._model is None:
            logger.info("Loading CrossEncoder model '%s' on device '%s'...", self._model_name, self._device)
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self._model_name,
                    device=self._device,
                    max_length=self._max_length,
                )
                logger.info("CrossEncoder model '%s' successfully loaded.", self._model_name)
            except Exception as exc:
                logger.error("Failed to load CrossEncoder model '%s': %s", self._model_name, exc)
                raise RuntimeError(f"Could not load CrossEncoder '{self._model_name}': {exc}") from exc

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: list[RetrievalCandidate],
        top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Rescore and reorder candidate passages against the query using joint cross-encoding.

        Args:
            query: The originating RetrievalQuery.
            candidates: Candidate passages (from Dense, Sparse, or Fusion) to be reranked.
            top_k: Maximum number of reranked candidates to return. If None, uses query.top_k.

        Returns:
            Ranked list of RetrievalCandidates sorted by cross-encoder relevance, sliced to top_k.
        """
        target_k = top_k if top_k is not None else query.top_k
        if target_k <= 0 or not candidates:
            return []

        # Ensure model is loaded
        if self._model is None:
            self.load()

        # Build input pairs (query_text, passage_text)
        q_text = query.text.strip()
        pairs: list[tuple[str, str]] = []

        for cand in candidates:
            p_text = cand.text.strip()
            # Truncation audit: check character length heuristic (~4 chars/token)
            # A passage > (max_length - 30) * 4 chars is at risk of truncation
            char_budget = (self._max_length - len(q_text.split()) - 10) * 4
            if len(p_text) > char_budget:
                self._truncation_count += 1
                logger.warning(
                    "Candidate '%s' length (%d chars) may exceed CrossEncoder max_length=%d.",
                    cand.record_id,
                    len(p_text),
                    self._max_length,
                )
            pairs.append((q_text, p_text))

        # Run model prediction
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.error("CrossEncoder prediction failed: %s", exc)
            raise RuntimeError(f"CrossEncoder prediction failed: {exc}") from exc

        # Create updated candidate records preserving all original provenance and branch scores
        scored_candidates: list[RetrievalCandidate] = []
        for i, cand in enumerate(candidates):
            raw_score = float(scores[i])

            # Deep-copy method scores and contributing methods to avoid mutation
            new_method_scores = dict(cand.method_scores)
            new_method_scores["rerank"] = raw_score

            new_contributing = list(cand.contributing_methods)
            if "rerank" not in new_contributing:
                new_contributing.append("rerank")

            new_metadata = dict(cand.metadata)
            new_metadata["pre_rerank_rank"] = cand.rank
            new_metadata["pre_rerank_score"] = cand.score

            scored_cand = RetrievalCandidate(
                record_id=cand.record_id,
                chunk_id=cand.chunk_id,
                rank=1,  # Temporary rank, will be finalized after sorting
                retrieval_method="rerank",
                score=raw_score,
                normalized_score=cand.normalized_score,
                text=cand.text,
                source_file=cand.source_file,
                source_pages=list(cand.source_pages),
                source_unit_ids=list(cand.source_unit_ids),
                section_title=cand.section_title,
                content_type=cand.content_type,
                fragment_index=cand.fragment_index,
                total_fragments=cand.total_fragments,
                contributing_methods=new_contributing,
                method_scores=new_method_scores,
                method_ranks=dict(cand.method_ranks),
                dense_score=cand.dense_score or cand.method_scores.get("dense"),
                sparse_score=cand.sparse_score or cand.method_scores.get("sparse"),
                rrf_score=cand.rrf_score or cand.method_scores.get("rrf"),
                rerank_score=raw_score,
                metadata=new_metadata,
            )
            scored_candidates.append(scored_cand)

        # Deterministic 3-tier tie-breaking:
        # Tier 1: -rerank_score (descending cross-encoder relevance)
        # Tier 2: -pre_rerank_score (descending original fusion/retrieval score)
        # Tier 3: record_id (ascending alphabetical tie-breaker)
        scored_candidates.sort(
            key=lambda c: (
                -c.score,
                -(c.metadata.get("pre_rerank_score") or 0.0),
                c.record_id,
            )
        )

        # Assign finalized 1-based ranks
        for rank_idx, c in enumerate(scored_candidates, start=1):
            c.rank = rank_idx
            c.method_ranks["rerank"] = rank_idx

        return scored_candidates[:target_k]
