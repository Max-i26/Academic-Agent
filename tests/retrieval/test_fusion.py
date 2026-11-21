"""Unit tests for ReciprocalRankFusion (RRF) algorithm and tie-breaking mechanics."""

import pytest
import numpy as np

from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    Evidence,
)
from academic_agent.retrieval.protocols import FusionStrategyProtocol
from academic_agent.retrieval.fusion import ReciprocalRankFusion


def make_candidate(
    record_id: str,
    rank: int,
    method: str,
    score: float,
    chunk_id: str | None = None,
    fragment_index: int = 0,
    total_fragments: int = 1,
    source_unit_ids: list[str] | None = None,
) -> RetrievalCandidate:
    """Helper to quickly construct synthetic candidates for fusion testing."""
    return RetrievalCandidate(
        record_id=record_id,
        chunk_id=chunk_id or record_id,
        rank=rank,
        retrieval_method=method,
        score=score,
        text=f"Sample text for {record_id}",
        source_file="Lecture4.pdf",
        source_pages=[7],
        source_unit_ids=source_unit_ids or [],
        section_title="Heading",
        content_type="PARAGRAPH",
        fragment_index=fragment_index,
        total_fragments=total_fragments,
    )


class TestReciprocalRankFusion:
    """Tests covering RRF mathematical formula, agreement, and edge cases."""

    def test_fusion_satisfies_protocol(self):
        rrf = ReciprocalRankFusion(k=60)
        assert isinstance(rrf, FusionStrategyProtocol)
        assert rrf.name == "rrf"
        assert rrf.k == 60

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            ReciprocalRankFusion(k=0)
        with pytest.raises(ValueError, match="k must be >= 1"):
            ReciprocalRankFusion(k=-10)

    def test_rrf_scoring_math_and_agreement(self):
        rrf = ReciprocalRankFusion(k=60)
        query = RetrievalQuery(text="SES formula", top_k=5)

        # Candidate A: Dense Rank 1, Sparse Rank 5
        # Candidate B: Dense Rank 2, Sparse Rank 1
        # Candidate C: Dense Rank 3, Missing in Sparse
        dense_cands = [
            make_candidate("doc_A", rank=1, method="dense", score=0.95),
            make_candidate("doc_B", rank=2, method="dense", score=0.85),
            make_candidate("doc_C", rank=3, method="dense", score=0.75),
        ]
        sparse_cands = [
            make_candidate("doc_B", rank=1, method="sparse", score=18.5),
            make_candidate("doc_A", rank=5, method="sparse", score=8.2),
        ]

        fused = rrf.fuse(
            query=query,
            candidates_by_method={"dense": dense_cands, "sparse": sparse_cands},
            top_k=3,
        )

        assert len(fused) == 3

        # Theoretical RRF Scores:
        # B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # A: 1/(60+1) + 1/(60+5) = 1/61 + 1/65 = 0.016393 + 0.015385 = 0.031778
        # C: 1/(60+3) = 1/63 = 0.015873
        expected_score_B = (1.0 / 62.0) + (1.0 / 61.0)
        expected_score_A = (1.0 / 61.0) + (1.0 / 65.0)
        expected_score_C = 1.0 / 63.0

        # Rank 1: Doc B
        assert fused[0].record_id == "doc_B"
        assert fused[0].rank == 1
        assert fused[0].retrieval_method == "hybrid"
        assert np.isclose(fused[0].score, expected_score_B, atol=1e-6)
        assert set(fused[0].contributing_methods) == {"dense", "sparse"}
        assert fused[0].method_scores == {"dense": 0.85, "sparse": 18.5}
        assert fused[0].method_ranks == {"dense": 2, "sparse": 1}

        # Rank 2: Doc A
        assert fused[1].record_id == "doc_A"
        assert fused[1].rank == 2
        assert np.isclose(fused[1].score, expected_score_A, atol=1e-6)
        assert set(fused[1].contributing_methods) == {"dense", "sparse"}
        assert fused[1].method_scores == {"dense": 0.95, "sparse": 8.2}

        # Rank 3: Doc C
        assert fused[2].record_id == "doc_C"
        assert fused[2].rank == 3
        assert np.isclose(fused[2].score, expected_score_C, atol=1e-6)
        assert fused[2].contributing_methods == ["dense"]
        assert fused[2].method_scores == {"dense": 0.75}

    def test_evidence_conversion_preserves_method_scores(self):
        rrf = ReciprocalRankFusion(k=60)
        query = RetrievalQuery(text="test", top_k=1)
        cand = make_candidate("doc_1", rank=1, method="dense", score=0.9, source_unit_ids=["u1"])
        cand_sp = make_candidate("doc_1", rank=2, method="sparse", score=14.0, source_unit_ids=["u1"])

        fused = rrf.fuse(
            query=query,
            candidates_by_method={"dense": [cand], "sparse": [cand_sp]},
            top_k=1,
        )
        assert len(fused) == 1

        ev = fused[0].to_evidence()
        assert isinstance(ev, Evidence)
        assert ev.record_id == "doc_1"
        assert set(ev.retrieval_methods) == {"dense", "sparse"}
        assert ev.method_scores == {"dense": 0.9, "sparse": 14.0}
        assert ev.source_unit_ids == ["u1"]

    def test_fragment_identity_preservation(self):
        """Verifies that fragments sharing a parent chunk ID are NOT merged together."""
        rrf = ReciprocalRankFusion(k=60)
        query = RetrievalQuery(text="fragment test", top_k=5)

        # Parent chunk 'chunk_parent' has 3 fragments
        dense_cands = [
            make_candidate("chunk_parent#frag_0", rank=1, method="dense", score=0.9, chunk_id="chunk_parent", fragment_index=0, total_fragments=3),
            make_candidate("chunk_parent#frag_1", rank=2, method="dense", score=0.8, chunk_id="chunk_parent", fragment_index=1, total_fragments=3),
        ]
        sparse_cands = [
            make_candidate("chunk_parent#frag_1", rank=1, method="sparse", score=15.0, chunk_id="chunk_parent", fragment_index=1, total_fragments=3),
            make_candidate("chunk_parent#frag_2", rank=2, method="sparse", score=12.0, chunk_id="chunk_parent", fragment_index=2, total_fragments=3),
        ]

        fused = rrf.fuse(
            query=query,
            candidates_by_method={"dense": dense_cands, "sparse": sparse_cands},
            top_k=3,
        )

        fused_ids = [c.record_id for c in fused]
        # Frag 1 appears in both (Rank 2 dense + Rank 1 sparse) -> highest RRF score
        assert fused[0].record_id == "chunk_parent#frag_1"
        assert fused[0].is_fragment is True
        assert set(fused[0].contributing_methods) == {"dense", "sparse"}

        # Frag 0 and Frag 2 remain separate independent candidates
        assert "chunk_parent#frag_0" in fused_ids
        assert "chunk_parent#frag_2" in fused_ids

    def test_duplicate_candidate_inside_single_branch_handled_safely(self):
        """Defensive test: malformed branch containing duplicate record_id keeps best rank and does not double-count."""
        rrf = ReciprocalRankFusion(k=60)
        query = RetrievalQuery(text="dup test", top_k=5)

        # Dense list contains duplicate 'doc_1' at rank 1 and rank 4
        dense_cands = [
            make_candidate("doc_1", rank=1, method="dense", score=0.9),
            make_candidate("doc_2", rank=2, method="dense", score=0.8),
            make_candidate("doc_1", rank=4, method="dense", score=0.6),  # duplicate!
        ]

        fused = rrf.fuse(
            query=query,
            candidates_by_method={"dense": dense_cands},
            top_k=5,
        )

        assert len(fused) == 2
        assert fused[0].record_id == "doc_1"
        # Score must be 1/(60+1), NOT 1/(60+1) + 1/(60+4)
        expected_score = 1.0 / 61.0
        assert np.isclose(fused[0].score, expected_score, atol=1e-6)

    def test_deterministic_tie_breaking(self):
        """Tests 4-tier tie breaking:
        1. RRF score
        2. Contributing method count
        3. Best individual rank
        4. Lexicographical record_id
        """
        rrf = ReciprocalRankFusion(k=60)
        query = RetrievalQuery(text="tie test", top_k=5)

        # Case: Same RRF score, but candidate X has 2 methods vs candidate Y with 1 method
        # To get same score: X in both at lower ranks vs Y in one at higher rank
        # 1/(60+3) = 1/63 = 0.0158730
        # For equal score test: 2 candidates with identical RRF score and agreement count,
        # but different best individual rank:
        # Candidate alpha: Rank 2 dense + Rank 4 sparse -> 1/62 + 1/64 = 0.016129 + 0.015625 = 0.031754
        # Candidate beta:  Rank 3 dense + Rank 3 sparse -> 1/63 + 1/63 = 0.015873 + 0.015873 = 0.031746

        # Direct tie test: Two candidates with identical scores and single method
        # Doc_B at rank 1 vs Doc_A at rank 1 in single method -> ties broken by record_id ("Doc_A" < "Doc_B")
        cands = [
            make_candidate("Doc_B", rank=1, method="dense", score=0.9),
            make_candidate("Doc_A", rank=1, method="dense", score=0.9),
        ]
        fused = rrf.fuse(query=query, candidates_by_method={"dense": cands}, top_k=2)
        assert fused[0].record_id == "Doc_A"
        assert fused[1].record_id == "Doc_B"

    def test_empty_branches_and_edge_cases(self):
        rrf = ReciprocalRankFusion(k=60)
        query = RetrievalQuery(text="empty test", top_k=5)

        # 1. Completely empty dict
        assert rrf.fuse(query, {}) == []

        # 2. Dict with empty lists
        assert rrf.fuse(query, {"dense": [], "sparse": []}) == []

        # 3. One populated branch, one empty branch
        single_cand = [make_candidate("doc_1", rank=1, method="sparse", score=10.0)]
        fused = rrf.fuse(query, {"dense": [], "sparse": single_cand}, top_k=5)
        assert len(fused) == 1
        assert fused[0].record_id == "doc_1"
        assert np.isclose(fused[0].score, 1.0 / 61.0, atol=1e-6)

        # 4. Top_k truncation
        cands = [make_candidate(f"doc_{i}", rank=i, method="dense", score=1.0 - i * 0.1) for i in range(1, 10)]
        fused_top2 = rrf.fuse(query, {"dense": cands}, top_k=2)
        assert len(fused_top2) == 2
        assert [c.rank for c in fused_top2] == [1, 2]
