"""Evidence context builder assembling token-budgeted, deduplicated evidence packages."""

import logging
import unicodedata
from typing import Any

from academic_agent.generation.synthesis.context.evidence_formatter import EvidenceFormatter
from academic_agent.generation.synthesis.models import (
    EvidenceContext,
    GroundedEvidence,
    OmissionReason,
    OmittedCandidate,
)
from academic_agent.generation.synthesis.protocols import (
    EvidenceContextBuilderProtocol,
    EvidenceFormatterProtocol,
)
from academic_agent.retrieval.models import Evidence, RetrievalResponse

logger = logging.getLogger(__name__)


class EvidenceContextBuilder(EvidenceContextBuilderProtocol):
    """Builds a budgeted, deduplicated, and safely ordered EvidenceContext from retrieval results.

    Key responsibilities:
    1. Retains ranking order from Phase 4 retrieval results by default.
    2. Case-preserving text and mathematical normalization (Unicode NFC, whitespace normalized).
    3. Sequence-aware overlap deduplication using n-gram word shingles (default 5-gram).
    4. Provenance union safeguard: merges page/unit metadata only when source_file is identical.
    5. Rank-monotonic budget packing with full-package XML token accounting.
    6. Complete audit logging of omitted candidates.
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        formatter: EvidenceFormatterProtocol | None = None,
        overlap_threshold: float = 0.85,
        min_overlap_words: int = 15,
        shingle_size: int = 5,
    ) -> None:
        if not (0.0 <= overlap_threshold <= 1.0):
            raise ValueError(f"overlap_threshold must be between 0.0 and 1.0, got {overlap_threshold}")
        if min_overlap_words < 1:
            raise ValueError(f"min_overlap_words must be at least 1, got {min_overlap_words}")
        if shingle_size < 1:
            raise ValueError(f"shingle_size must be at least 1, got {shingle_size}")

        self.tokenizer = tokenizer
        self.formatter: EvidenceFormatterProtocol = formatter or EvidenceFormatter()
        self.overlap_threshold = overlap_threshold
        self.min_overlap_words = min_overlap_words
        self.shingle_size = shingle_size

    @staticmethod
    def normalize_text(text: str) -> str:
        """Case-preserving Unicode NFC and whitespace normalization.

        Preserves case distinctions critical in mathematics (e.g. 'X' vs 'x', 'ACF' vs 'acf', 'MA(1)' vs 'ma(1)').
        """
        nfc = unicodedata.normalize("NFC", text)
        return " ".join(nfc.split())

    @staticmethod
    def get_shingles(words: list[str], n: int = 5) -> list[tuple[str, ...]]:
        """Extract contiguous n-gram word shingles preserving sequence order."""
        if len(words) < n:
            return []
        return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    def count_tokens(self, text: str) -> tuple[int, bool]:
        """Count tokens using injected tokenizer if available, or conservative heuristic fallback.

        Returns:
            Tuple of (token_count, is_exact_token_count).
        """
        if not text:
            return 0, (self.tokenizer is not None)

        if self.tokenizer is not None:
            try:
                if callable(self.tokenizer):
                    result = self.tokenizer(text)
                    if isinstance(result, int):
                        return result, True
                    if isinstance(result, (list, tuple)):
                        return len(result), True
                    if isinstance(result, dict) and "input_ids" in result:
                        return len(result["input_ids"]), True
                if hasattr(self.tokenizer, "encode"):
                    encoded = self.tokenizer.encode(text)
                    if isinstance(encoded, (list, tuple)):
                        return len(encoded), True
                    if isinstance(encoded, dict) and "input_ids" in encoded:
                        return len(encoded["input_ids"]), True
            except Exception as e:
                logger.warning(f"Injected tokenizer failed on text ({e}); falling back to heuristic.")

        # Heuristic fallback: conservative bounds for academic and LaTeX text
        words = len(text.split())
        chars = len(text)
        est_tokens = max(1, max(int(chars / 2.8), int(words * 1.35)) + 4)
        return est_tokens, False

    def build_context(
        self,
        retrieval_response: RetrievalResponse,
        allocated_evidence_tokens: int,
    ) -> EvidenceContext:
        """Assemble a budgeted evidence context from retrieval results.

        Args:
            retrieval_response: Results surfaced by Phase 4 Retrieval Engine.
            allocated_evidence_tokens: Exact token ceiling allocated to evidence package.

        Returns:
            EvidenceContext respecting the budget allocation and audit requirements.
        """
        if allocated_evidence_tokens < 0:
            raise ValueError(
                f"allocated_evidence_tokens cannot be negative, got {allocated_evidence_tokens}"
            )

        candidates: list[Evidence] = retrieval_response.evidence or []
        is_exact = self.tokenizer is not None

        if not candidates:
            return EvidenceContext(
                items=[],
                total_tokens=0,
                budget_allocated=allocated_evidence_tokens,
                omitted_candidates=[],
                is_exact_token_count=is_exact,
            )

        if allocated_evidence_tokens == 0:
            omitted = [
                OmittedCandidate(
                    record_id=cand.record_id,
                    chunk_id=cand.chunk_id,
                    source_file=cand.source_file,
                    source_pages=cand.source_pages,
                    reason=OmissionReason.BUDGET_EXCEEDED,
                    details="Allocated evidence budget is 0 tokens",
                    rank=idx + 1,
                )
                for idx, cand in enumerate(candidates)
            ]
            return EvidenceContext(
                items=[],
                total_tokens=0,
                budget_allocated=0,
                omitted_candidates=omitted,
                is_exact_token_count=is_exact,
            )

        accepted: list[GroundedEvidence] = []
        omitted: list[OmittedCandidate] = []
        current_package_tokens = 0

        for idx, cand in enumerate(candidates):
            rank = idx + 1
            raw_text = cand.text or ""
            norm_cand_text = self.normalize_text(raw_text)

            # Check 1: Non-empty valid text
            if not norm_cand_text:
                omitted.append(
                    OmittedCandidate(
                        record_id=cand.record_id,
                        chunk_id=cand.chunk_id,
                        source_file=cand.source_file,
                        source_pages=cand.source_pages,
                        reason=OmissionReason.INVALID_CANDIDATE,
                        details="Empty or whitespace-only candidate text",
                        rank=rank,
                    )
                )
                continue

            cand_words = norm_cand_text.split()
            cand_shingles = self.get_shingles(cand_words, n=self.shingle_size)
            cand_shingles_set = set(cand_shingles)

            # Check 2: Exact duplicates & record_id integrity checks against accepted items
            is_duplicate = False
            for acc_idx, acc_item in enumerate(accepted):
                acc_norm_text = self.normalize_text(acc_item.text)

                # Check 2a: Matching record_id
                if cand.record_id == acc_item.evidence.record_id:
                    if cand.source_file == acc_item.source_file and norm_cand_text == acc_norm_text:
                        # Genuine duplicate record with identical source file and text -> merge pages
                        original_ge = acc_item
                        merged_pages = sorted(list(set(acc_item.source_pages) | set(cand.source_pages)))
                        merged_units = sorted(list(set(acc_item.evidence.source_unit_ids) | set(cand.source_unit_ids)))
                        updated_ev = acc_item.evidence.model_copy(
                            update={"source_pages": merged_pages, "source_unit_ids": merged_units}
                        )
                        updated_ge = GroundedEvidence(
                            evidence_id=acc_item.evidence_id,
                            evidence=updated_ev,
                            citation_label=updated_ev.format_citation(),
                            token_count=acc_item.token_count,
                            is_truncated=acc_item.is_truncated,
                        )
                        item_xml = self.formatter.format_item(updated_ge)
                        cnt, c_exact = self.count_tokens(item_xml)
                        updated_ge = updated_ge.model_copy(update={"token_count": cnt})
                        if not c_exact:
                            is_exact = False

                        # Tentatively test merged package budget
                        accepted[acc_idx] = updated_ge
                        pkg_xml = self.formatter.format_evidence_block(EvidenceContext(items=accepted))
                        merged_package_tokens, pkg_exact = self.count_tokens(pkg_xml)
                        if not pkg_exact:
                            is_exact = False

                        if merged_package_tokens > allocated_evidence_tokens:
                            # Rollback merge: page expansion would exceed budget
                            accepted[acc_idx] = original_ge
                            omitted.append(
                                OmittedCandidate(
                                    record_id=cand.record_id,
                                    chunk_id=cand.chunk_id,
                                    source_file=cand.source_file,
                                    source_pages=cand.source_pages,
                                    reason=OmissionReason.BUDGET_EXCEEDED,
                                    details=f"Provenance merge would expand evidence package tokens ({merged_package_tokens}) beyond budget ({allocated_evidence_tokens})",
                                    rank=rank,
                                )
                            )
                            is_duplicate = True
                            break

                        current_package_tokens = merged_package_tokens
                        accepted[acc_idx] = updated_ge

                        omitted.append(
                            OmittedCandidate(
                                record_id=cand.record_id,
                                chunk_id=cand.chunk_id,
                                source_file=cand.source_file,
                                source_pages=cand.source_pages,
                                reason=OmissionReason.DUPLICATE,
                                details=f"Identical record_id '{cand.record_id}' merged into {acc_item.evidence_id}",
                                rank=rank,
                            )
                        )
                        is_duplicate = True
                        break
                    else:
                        # Conflicting source_file or content for the same record_id -> flag integrity anomaly
                        logger.warning(
                            f"Integrity anomaly: candidate {cand.record_id} has conflicting content or source file."
                        )
                        omitted.append(
                            OmittedCandidate(
                                record_id=cand.record_id,
                                chunk_id=cand.chunk_id,
                                source_file=cand.source_file,
                                source_pages=cand.source_pages,
                                reason=OmissionReason.INVALID_CANDIDATE,
                                details=f"Conflicting record_id '{cand.record_id}' with different source file or content; omitted for integrity",
                                rank=rank,
                            )
                        )
                        is_duplicate = True
                        break

                # Check 2b: Matching text from identical source file
                if norm_cand_text == acc_norm_text and cand.source_file == acc_item.source_file:
                    original_ge = acc_item
                    merged_pages = sorted(list(set(acc_item.source_pages) | set(cand.source_pages)))
                    merged_units = sorted(list(set(acc_item.evidence.source_unit_ids) | set(cand.source_unit_ids)))
                    updated_ev = acc_item.evidence.model_copy(
                        update={"source_pages": merged_pages, "source_unit_ids": merged_units}
                    )
                    updated_ge = GroundedEvidence(
                        evidence_id=acc_item.evidence_id,
                        evidence=updated_ev,
                        citation_label=updated_ev.format_citation(),
                        token_count=acc_item.token_count,
                        is_truncated=acc_item.is_truncated,
                    )
                    item_xml = self.formatter.format_item(updated_ge)
                    cnt, c_exact = self.count_tokens(item_xml)
                    updated_ge = updated_ge.model_copy(update={"token_count": cnt})
                    if not c_exact:
                        is_exact = False

                    # Tentatively test merged package budget
                    accepted[acc_idx] = updated_ge
                    pkg_xml = self.formatter.format_evidence_block(EvidenceContext(items=accepted))
                    merged_package_tokens, pkg_exact = self.count_tokens(pkg_xml)
                    if not pkg_exact:
                        is_exact = False

                    if merged_package_tokens > allocated_evidence_tokens:
                        # Rollback merge: page expansion would exceed budget
                        accepted[acc_idx] = original_ge
                        omitted.append(
                            OmittedCandidate(
                                record_id=cand.record_id,
                                chunk_id=cand.chunk_id,
                                source_file=cand.source_file,
                                source_pages=cand.source_pages,
                                reason=OmissionReason.BUDGET_EXCEEDED,
                                details=f"Provenance merge would expand evidence package tokens ({merged_package_tokens}) beyond budget ({allocated_evidence_tokens})",
                                rank=rank,
                            )
                        )
                        is_duplicate = True
                        break

                    current_package_tokens = merged_package_tokens
                    accepted[acc_idx] = updated_ge

                    omitted.append(
                        OmittedCandidate(
                            record_id=cand.record_id,
                            chunk_id=cand.chunk_id,
                            source_file=cand.source_file,
                            source_pages=cand.source_pages,
                            reason=OmissionReason.DUPLICATE,
                            details=f"Identical text from same source file '{cand.source_file}' merged into {acc_item.evidence_id}",
                            rank=rank,
                        )
                    )
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # Check 3: Sequence-aware n-gram overlap within same source file
            is_overlapping = False
            for acc_idx, acc_item in enumerate(accepted):
                # Merging provenance is only allowed when source_file is identical
                if cand.source_file != acc_item.source_file:
                    continue

                acc_norm_text = self.normalize_text(acc_item.text)
                acc_words = acc_norm_text.split()

                if len(cand_words) >= self.min_overlap_words and len(acc_words) >= self.min_overlap_words:
                    acc_shingles = self.get_shingles(acc_words, n=self.shingle_size)
                    acc_shingles_set = set(acc_shingles)

                    ratio_cand = (
                        sum(1 for s in cand_shingles if s in acc_shingles_set) / len(cand_shingles)
                        if cand_shingles
                        else 0.0
                    )
                    ratio_acc = (
                        sum(1 for s in acc_shingles if s in cand_shingles_set) / len(acc_shingles)
                        if acc_shingles
                        else 0.0
                    )
                    overlap_ratio = max(ratio_cand, ratio_acc)

                    if overlap_ratio >= self.overlap_threshold:
                        original_ge = acc_item
                        merged_pages = sorted(list(set(acc_item.source_pages) | set(cand.source_pages)))
                        merged_units = sorted(list(set(acc_item.evidence.source_unit_ids) | set(cand.source_unit_ids)))
                        updated_ev = acc_item.evidence.model_copy(
                            update={"source_pages": merged_pages, "source_unit_ids": merged_units}
                        )
                        updated_ge = GroundedEvidence(
                            evidence_id=acc_item.evidence_id,
                            evidence=updated_ev,
                            citation_label=updated_ev.format_citation(),
                            token_count=acc_item.token_count,
                            is_truncated=acc_item.is_truncated,
                        )
                        item_xml = self.formatter.format_item(updated_ge)
                        cnt, c_exact = self.count_tokens(item_xml)
                        updated_ge = updated_ge.model_copy(update={"token_count": cnt})
                        if not c_exact:
                            is_exact = False

                        # Tentatively test merged package budget
                        accepted[acc_idx] = updated_ge
                        pkg_xml = self.formatter.format_evidence_block(EvidenceContext(items=accepted))
                        merged_package_tokens, pkg_exact = self.count_tokens(pkg_xml)
                        if not pkg_exact:
                            is_exact = False

                        if merged_package_tokens > allocated_evidence_tokens:
                            # Rollback merge: page expansion would exceed budget
                            accepted[acc_idx] = original_ge
                            omitted.append(
                                OmittedCandidate(
                                    record_id=cand.record_id,
                                    chunk_id=cand.chunk_id,
                                    source_file=cand.source_file,
                                    source_pages=cand.source_pages,
                                    reason=OmissionReason.BUDGET_EXCEEDED,
                                    details=f"Provenance merge would expand evidence package tokens ({merged_package_tokens}) beyond budget ({allocated_evidence_tokens})",
                                    rank=rank,
                                )
                            )
                            is_overlapping = True
                            break

                        current_package_tokens = merged_package_tokens
                        accepted[acc_idx] = updated_ge

                        omitted.append(
                            OmittedCandidate(
                                record_id=cand.record_id,
                                chunk_id=cand.chunk_id,
                                source_file=cand.source_file,
                                source_pages=cand.source_pages,
                                reason=OmissionReason.OVERLAPPING,
                                details=f"Sequence containment {overlap_ratio:.1%} >= {self.overlap_threshold:.1%} with {acc_item.evidence_id} (pages merged)",
                                rank=rank,
                            )
                        )
                        is_overlapping = True
                        break

            if is_overlapping:
                continue

            # Check 4: Tentative Budget Packing (Rank-monotonic full package accounting)
            tentative_id = f"E{len(accepted) + 1}"
            tentative_ge = GroundedEvidence(
                evidence_id=tentative_id,
                evidence=cand,
            )
            tentative_xml = self.formatter.format_evidence_block(
                EvidenceContext(items=accepted + [tentative_ge])
            )
            tentative_tokens, c_exact1 = self.count_tokens(tentative_xml)
            if not c_exact1:
                is_exact = False

            if tentative_tokens <= allocated_evidence_tokens:
                # Candidate fits safely within budget
                item_xml = self.formatter.format_item(tentative_ge)
                t_cnt, c_exact2 = self.count_tokens(item_xml)
                tentative_ge = tentative_ge.model_copy(update={"token_count": t_cnt})
                if not c_exact2:
                    is_exact = False
                accepted.append(tentative_ge)
                current_package_tokens = tentative_tokens
            else:
                # Candidate exceeds budget -> halt packing to preserve rank monotonicity
                omitted.append(
                    OmittedCandidate(
                        record_id=cand.record_id,
                        chunk_id=cand.chunk_id,
                        source_file=cand.source_file,
                        source_pages=cand.source_pages,
                        reason=OmissionReason.BUDGET_EXCEEDED,
                        details=f"Package tokens ({tentative_tokens}) exceeds allocated budget ({allocated_evidence_tokens})",
                        rank=rank,
                    )
                )

                # Record all subsequent candidates as omitted due to budget exhaustion by higher-ranked items
                for rem_idx in range(idx + 1, len(candidates)):
                    rem_cand = candidates[rem_idx]
                    omitted.append(
                        OmittedCandidate(
                            record_id=rem_cand.record_id,
                            chunk_id=rem_cand.chunk_id,
                            source_file=rem_cand.source_file,
                            source_pages=rem_cand.source_pages,
                            reason=OmissionReason.BUDGET_EXCEEDED,
                            details="Omitted after budget exhausted by higher-ranked candidates",
                            rank=rem_idx + 1,
                        )
                    )
                break

        final_tokens = current_package_tokens if accepted else 0
        return EvidenceContext(
            items=accepted,
            total_tokens=final_tokens,
            budget_allocated=allocated_evidence_tokens,
            omitted_candidates=omitted,
            is_exact_token_count=is_exact,
        )
