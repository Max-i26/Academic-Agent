"""Academic search tool wrapping Phase 4 hybrid retrieval and reranking."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from academic_agent.agent.models import ToolSideEffectPolicy
from academic_agent.agent.tools.base import BaseAgentTool
from academic_agent.retrieval.models import RetrievalQuery
from academic_agent.retrieval.protocols import RetrievalPipelineProtocol
from academic_agent.storage.models import FilterQuery


class AcademicSearchArgs(BaseModel):
    """Input parameters for academic search."""

    query: str = Field(min_length=1, description="Academic question, concepts, or formulas to search for.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of relevant evidence passages to retrieve.")
    course_id: Optional[str] = Field(default=None, description="Optional filter scoping search to a specific course ID.")
    lecture_id: Optional[str] = Field(default=None, description="Optional filter scoping search to a specific lecture file.")


class AcademicSearchTool(BaseAgentTool):
    """Retrieves grounded lecture evidence using hybrid dense + sparse search with cross-encoder reranking."""

    name = "academic_search"
    description = (
        "Search academic lecture slides and course materials using hybrid BM25 lexical "
        "and dense vector semantic retrieval with neural cross-encoder reranking."
    )
    side_effect_policy = ToolSideEffectPolicy(
        read_only=True,
        idempotent=True,
        mutating=False,
        requires_confirmation=False,
        timeout_seconds=15.0,
    )
    args_model = AcademicSearchArgs

    def __init__(self, pipeline: RetrievalPipelineProtocol) -> None:
        self.pipeline = pipeline

    def _run(self, args: AcademicSearchArgs) -> Dict[str, Any]:
        """Execute hybrid search pipeline with metadata filtering and provenance capture."""
        filter_obj: Optional[FilterQuery] = None
        if args.course_id or args.lecture_id:
            custom = {}
            if args.course_id:
                custom["course_id"] = args.course_id
            filter_obj = FilterQuery(
                source_file=args.lecture_id,
                custom_filters=custom if custom else None,
            )

        retrieval_query = RetrievalQuery(
            text=args.query,
            top_k=args.top_k,
            filters=filter_obj,
        )

        if hasattr(self.pipeline, "retrieve"):
            response = self.pipeline.retrieve(retrieval_query)
        elif hasattr(self.pipeline, "search"):
            response = self.pipeline.search(retrieval_query)
        else:
            raise AttributeError(f"Pipeline {type(self.pipeline)} implements neither 'retrieve' nor 'search'.")

        candidates: List[Dict[str, Any]] = []
        for ev in response.evidence:
            candidates.append({
                "record_id": ev.record_id,
                "chunk_id": ev.chunk_id,
                "text": ev.text,
                "source_file": ev.source_file,
                "source_pages": ev.source_pages,
                "score": round(float(ev.score), 4),
                "citation": ev.format_citation() if hasattr(ev, "format_citation") else f"[{ev.source_file}]",
                "metadata": ev.metadata,
            })

        return {
            "query": args.query,
            "total_found": len(candidates),
            "candidates": candidates,
        }
