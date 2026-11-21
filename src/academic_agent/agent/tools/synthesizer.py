"""Academic answer synthesis tool wrapping Phase 5 AcademicSynthesizer."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from academic_agent.agent.models import ToolSideEffectPolicy
from academic_agent.agent.tools.base import BaseAgentTool
from academic_agent.generation.synthesis.models import AcademicAnswer, PolicyOutcome
from academic_agent.generation.synthesis.protocols import AcademicSynthesizerProtocol
from academic_agent.retrieval.models import Evidence, RetrievalDiagnostics, RetrievalQuery, RetrievalResponse


class AcademicSynthesizerArgs(BaseModel):
    """Input parameters for academic answer synthesis."""

    query: str = Field(min_length=1, description="Student's academic question.")
    evidence_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Candidate evidence passages surfaced and filtered by previous tool calls.",
    )
    allocated_evidence_tokens: Optional[int] = Field(
        default=None,
        ge=50,
        description="Optional evidence token budget ceiling passed to context builder.",
    )


class AcademicSynthesizerTool(BaseAgentTool):
    """Executes grounded synthesis with citation verification, code masking, and policy validation."""

    name = "academic_synthesizer"
    description = (
        "Generate a grounded academic answer from verified lecture evidence with strict inline "
        "citation extraction, non-destructive reference resolution, and policy enforcement."
    )
    side_effect_policy = ToolSideEffectPolicy(
        read_only=True,
        idempotent=True,
        mutating=False,
        requires_confirmation=False,
        timeout_seconds=30.0,
    )
    args_model = AcademicSynthesizerArgs

    def __init__(self, synthesizer: AcademicSynthesizerProtocol) -> None:
        self.synthesizer = synthesizer

    def _run(self, args: AcademicSynthesizerArgs) -> Dict[str, Any]:
        """Convert input candidates to RetrievalResponse, invoke synthesizer, and format audit observation."""
        evidence_list: List[Evidence] = []
        for idx, cand in enumerate(args.evidence_candidates):
            cid = cand.get("record_id") or cand.get("chunk_id") or f"cand_{idx}"
            chunk_id = cand.get("chunk_id") or cid
            text = cand.get("text") or cand.get("content") or ""
            source_file = cand.get("source_file") or "lecture.pdf"
            source_pages = cand.get("source_pages") or [1]
            score = float(cand.get("score", 1.0))
            meta = cand.get("metadata") or {}

            ev = Evidence(
                record_id=cid,
                chunk_id=chunk_id,
                text=text,
                source_file=source_file,
                source_pages=source_pages,
                score=score,
                metadata=meta,
            )
            evidence_list.append(ev)

        rq = RetrievalQuery(
            text=args.query,
            top_k=max(1, len(evidence_list)),
        )

        retrieval_response = RetrievalResponse(
            query=rq,
            evidence=evidence_list,
            diagnostics=RetrievalDiagnostics(retrieval_strategy="agent_orchestrated"),
        )

        answer: AcademicAnswer = self.synthesizer.synthesize(
            query=args.query,
            retrieval_response=retrieval_response,
            allocated_evidence_tokens=args.allocated_evidence_tokens,
        )

        outcome = answer.policy_outcome
        outcome_val = outcome.value if hasattr(outcome, "value") else str(outcome)
        citations = getattr(answer.validation, "citations", [])
        omissions = getattr(answer.evidence_context, "omitted_candidates", [])

        return {
            "answer_text": answer.resolved_answer_text or answer.answer_text,
            "raw_answer": answer.answer_text,
            "policy_outcome": outcome_val,
            "is_policy_compliant": answer.is_policy_compliant,
            "citations_count": len(citations),
            "citations": [c.model_dump() for c in citations],
            "valid_cited_ids": list(answer.validation.valid_cited_ids),
            "invalid_cited_ids": list(answer.validation.invalid_cited_ids),
            "omissions_count": len(omissions),
            "omissions": [o.model_dump() for o in omissions],
            "latency_ms": answer.latency_ms,
            "is_valid_academic_answer": outcome in {PolicyOutcome.ANSWERED, PolicyOutcome.PARTIAL_ANSWER},
        }
