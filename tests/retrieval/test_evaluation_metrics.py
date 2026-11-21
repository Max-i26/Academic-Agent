"""Unit tests verifying mathematical accuracy of evaluation metrics against synthetic rankings."""

import math
import numpy as np
import pytest

from academic_agent.retrieval.evaluator import (
    compute_hit_at_k,
    compute_reciprocal_rank,
    compute_average_precision,
    compute_recall_at_k,
    compute_ndcg_at_k,
    get_record_relevance,
)


class TestEvaluationMetricsMath:
    """Verifies each metric against mathematically pre-calculated synthetic rankings."""

    def test_hit_at_k_exact_behavior(self):
        relevant = {"doc_rel": 3}

        # Case 1: At rank 1
        ranked_1 = ["doc_rel", "doc_dist1", "doc_dist2"]
        assert compute_hit_at_k(ranked_1, relevant, k=1) == 1.0
        assert compute_hit_at_k(ranked_1, relevant, k=3) == 1.0

        # Case 2: At rank 2
        ranked_2 = ["doc_dist1", "doc_rel", "doc_dist2"]
        assert compute_hit_at_k(ranked_2, relevant, k=1) == 0.0
        assert compute_hit_at_k(ranked_2, relevant, k=2) == 1.0
        assert compute_hit_at_k(ranked_2, relevant, k=3) == 1.0

        # Case 3: At rank 4
        ranked_4 = ["d1", "d2", "d3", "doc_rel", "d5"]
        assert compute_hit_at_k(ranked_4, relevant, k=3) == 0.0
        assert compute_hit_at_k(ranked_4, relevant, k=4) == 1.0
        assert compute_hit_at_k(ranked_4, relevant, k=5) == 1.0

        # Case 4: Not in top k
        assert compute_hit_at_k(["d1", "d2"], relevant, k=2) == 0.0
        assert compute_hit_at_k([], relevant, k=5) == 0.0

    def test_reciprocal_rank_exact_math(self):
        relevant = {"target": 2}

        # Rank 1 -> 1.0
        assert compute_reciprocal_rank(["target", "d1"], relevant) == 1.0
        # Rank 2 -> 0.5
        assert compute_reciprocal_rank(["d1", "target"], relevant) == 0.5
        # Rank 4 -> 0.25
        assert compute_reciprocal_rank(["d1", "d2", "d3", "target"], relevant) == 0.25
        # Missing -> 0.0
        assert compute_reciprocal_rank(["d1", "d2"], relevant) == 0.0

    def test_average_precision_exact_math(self):
        # Case A: Two relevant documents in corpus: doc_A, doc_B
        relevant = {"doc_A": 3, "doc_B": 2}  # R = 2

        # 1. Perfect ranking: [doc_A, doc_B, d1]
        # P@1 = 1/1 = 1.0; P@2 = 2/2 = 1.0 -> AP = (1.0 + 1.0) / 2 = 1.0
        assert np.isclose(compute_average_precision(["doc_A", "doc_B", "d1"], relevant), 1.0)

        # 2. Ranked at 1 and 3: [doc_A, d1, doc_B]
        # P@1 = 1/1 = 1.0
        # P@2 = 1/2 = 0.5 (non-relevant)
        # P@3 = 2/3 = 0.6667
        # AP = (1.0 + 2/3) / 2 = 5/6 = 0.83333
        expected_ap = (1.0 + 2.0 / 3.0) / 2.0
        assert np.isclose(compute_average_precision(["doc_A", "d1", "doc_B"], relevant), expected_ap, atol=1e-5)

        # 3. Only one relevant retrieved at rank 1: [doc_A, d1, d2]
        # P@1 = 1/1 = 1.0
        # AP = 1.0 / 2 = 0.5
        assert np.isclose(compute_average_precision(["doc_A", "d1", "d2"], relevant), 0.5)

        # 4. Neither retrieved: [d1, d2] -> AP = 0.0
        assert compute_average_precision(["d1", "d2"], relevant) == 0.0

    def test_recall_at_k_exact_math(self):
        relevant = {"doc_1": 3, "doc_2": 2}  # R = 2
        ranked = ["doc_1", "d_dist", "d_dist2", "doc_2"]

        # At k=1: 1 retrieved out of 2 -> 0.5
        assert compute_recall_at_k(ranked, relevant, k=1) == 0.5
        # At k=2: 1 retrieved out of 2 -> 0.5
        assert compute_recall_at_k(ranked, relevant, k=2) == 0.5
        # At k=4: 2 retrieved out of 2 -> 1.0
        assert compute_recall_at_k(ranked, relevant, k=4) == 1.0

    def test_ndcg_at_k_exact_math(self):
        # Two graded items: doc_3 (grade 3), doc_2 (grade 2)
        relevant = {"doc_3": 3, "doc_2": 2}

        # Ideal DCG@2:
        # Gain(3) = 2^3 - 1 = 7. Discount = log2(1+1) = 1. Term1 = 7/1 = 7.0
        # Gain(2) = 2^2 - 1 = 3. Discount = log2(2+1) = 1.5849625. Term2 = 3/1.5849625 = 1.892789
        # IDCG@2 = 7.0 + 1.892789 = 8.892789
        idcg_2 = (7.0 / 1.0) + (3.0 / math.log2(3.0))

        # 1. Perfect ranking: [doc_3, doc_2] -> nDCG = 1.0
        perfect_ranked = ["doc_3", "doc_2", "irrelevant"]
        assert np.isclose(compute_ndcg_at_k(perfect_ranked, relevant, k=2), 1.0, atol=1e-5)

        # 2. Inverted ranking: [doc_2, doc_3]
        # DCG@2 = (3/1) + (7 / log2(3)) = 3.0 + 4.416508 = 7.416508
        # nDCG@2 = 7.416508 / 8.892789 = 0.8340
        expected_inverted_ndcg = ((3.0 / 1.0) + (7.0 / math.log2(3.0))) / idcg_2
        assert np.isclose(compute_ndcg_at_k(["doc_2", "doc_3"], relevant, k=2), expected_inverted_ndcg, atol=1e-5)

        # 3. Irrelevant item at rank 1: [irrelevant, doc_3]
        # DCG@2 = 0 + (7 / log2(3)) = 4.416508
        # nDCG@2 = 4.416508 / 8.892789 = 0.4966
        expected_displaced_ndcg = (7.0 / math.log2(3.0)) / idcg_2
        assert np.isclose(compute_ndcg_at_k(["irrelevant", "doc_3"], relevant, k=2), expected_displaced_ndcg, atol=1e-5)

    def test_fragment_parent_tolerance(self):
        """Verifies that fragment IDs (e.g. chunk_xyz#frag_1) correctly match parent chunk chunk_xyz."""
        relevant = {"chunk_cma": 3}
        ranked = ["chunk_cma#frag_1", "distractor"]

        assert get_record_relevance("chunk_cma#frag_1", relevant) == 3
        assert compute_hit_at_k(ranked, relevant, k=1) == 1.0
        assert compute_reciprocal_rank(ranked, relevant, k=1) == 1.0
        assert np.isclose(compute_ndcg_at_k(ranked, relevant, k=1), 1.0)
