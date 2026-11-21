"""Comprehensive integration and trajectory tests for AcademicAgent orchestrator (Stage 6.4.7 & 6.4.8)."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from typing import Any, Dict, List, Optional
import pytest

from academic_agent.agent.budget import ExecutionBudgetTracker
from academic_agent.agent.context import (
    AgentExecutionContext,
    AgentRunConfig,
    CancellationToken,
)
from academic_agent.agent.decision_policy import DecisionPolicy
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import (
    ActionType,
    AgentExecutionState,
    AgentTerminalState,
    AgentTrace,
    BudgetConfig,
    ProposedAction,
    ToolCall,
    ToolErrorCategory,
    ToolResult,
    ToolSideEffectPolicy,
)
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.planner import ReActStepPlanner
from academic_agent.agent.registry import ToolRegistry
from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    MessageRole,
)
from academic_agent.generation.protocols import LLMProviderProtocol
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    CitationOccurrence,
    EvidenceContext,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)


# ============================================================================
# Test Doubles & Fixtures
# ============================================================================

class MockLLMProvider:
    """Mock LLM provider returning a sequence of scripted JSON responses."""

    def __init__(self, responses: List[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return "mock-react-llm"

    @property
    def context_window(self) -> int:
        return 4096

    @property
    def is_available(self) -> bool:
        return True

    def generate(self, messages: List[ChatMessage], config: Optional[LLMConfig] = None) -> LLMResponse:
        with self._lock:
            if self.call_count < len(self.responses):
                content = self.responses[self.call_count]
            else:
                content = self.responses[-1]
            self.call_count += 1

        return LLMResponse(
            content=content,
            role=MessageRole.ASSISTANT,
            model_name=self.model_name,
            latency_ms=25.0,
            usage=LLMUsage(prompt_tokens=120, completion_tokens=40, total_tokens=160),
        )

    def load(self) -> None:
        pass

    def release(self) -> None:
        pass


class MockAcademicSearchTool:
    def __init__(self, candidates: Optional[List[Dict[str, Any]]] = None) -> None:
        self.candidates = candidates if candidates is not None else [
            {
                "candidate_id": "cand_0",
                "source_file": "Lecture_04.pdf",
                "page": 12,
                "score": 0.88,
                "content": "Backpropagation computes gradients using the chain rule.",
            }
        ]

    @property
    def name(self) -> str:
        return "academic_search"

    @property
    def description(self) -> str:
        return "Search lecture repository"

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, idempotent=True, timeout_seconds=5.0)

    def get_schema(self) -> dict:
        return {
            "name": "academic_search",
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(
            call_id="call_s",
            tool_name="academic_search",
            success=True,
            output={"query": arguments.get("query", ""), "count": len(self.candidates), "candidates": self.candidates},
        )


class MockEvidenceFilterTool:
    @property
    def name(self) -> str:
        return "evidence_filter"

    @property
    def description(self) -> str:
        return "Filter candidates"

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, idempotent=True, timeout_seconds=5.0)

    def get_schema(self) -> dict:
        return {
            "name": "evidence_filter",
            "type": "object",
            "properties": {"candidates": {"type": "array"}},
            "required": [],
        }

    def execute(self, arguments: dict) -> ToolResult:
        cands = arguments.get("candidates", [])
        return ToolResult(
            call_id="call_f",
            tool_name="evidence_filter",
            success=True,
            output={"retained_candidates": cands, "omitted_candidates": [], "retained_count": len(cands), "omitted_count": 0},
        )


class MockAcademicSynthesizerTool:
    def __init__(self, is_valid: bool = True) -> None:
        self.is_valid = is_valid

    @property
    def name(self) -> str:
        return "academic_synthesizer"

    @property
    def description(self) -> str:
        return "Synthesize grounded answer"

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, idempotent=True, timeout_seconds=5.0)

    def get_schema(self) -> dict:
        return {
            "name": "academic_synthesizer",
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def execute(self, arguments: dict) -> ToolResult:
        query_text = arguments.get("query", "test")
        ans = AcademicAnswer(
            answer_text="Backpropagation applies chain rule recursively [1].",
            resolved_answer_text="Backpropagation applies chain rule recursively [Lecture 04, p.12].",
            query=query_text,
            evidence_context=EvidenceContext(),
            validation=ValidationDiagnostics(
                citations=[
                    CitationOccurrence(
                        evidence_id="cand_0",
                        raw_citation="[1]",
                        start_char=44,
                        end_char=47,
                        is_valid=True,
                        resolution_label="Lecture 04, p.12",
                    )
                ],
                valid_cited_ids=["cand_0"],
                invalid_cited_ids=[],
            ),
            policy_result=PolicyValidationResult(
                outcome=PolicyOutcome.ANSWERED if self.is_valid else PolicyOutcome.COMPLIANT_ABSTENTION,
                is_compliant=self.is_valid,
            ),
            raw_response=LLMResponse(
                content="Backpropagation applies chain rule recursively [1].",
                model_name="mock-synthesizer",
                latency_ms=10.0,
            ),
            model_name="mock-synthesizer",
            latency_ms=10.0,
        )
        return ToolResult(
            call_id="call_syn",
            tool_name="academic_synthesizer",
            success=True,
            output={
                "answer_text": ans.resolved_answer_text,
                "raw_answer": ans.answer_text,
                "is_valid_academic_answer": self.is_valid,
                "academic_answer": ans,
            },
        )


# ============================================================================
# Stage 6.4.7 — End-to-End Trajectory Tests
# ============================================================================

class TestOrchestratorEndToEndTrajectories:
    """Test full multi-step agent trajectories from query to termination."""

    def test_standard_successful_research_and_synthesis_trajectory(self) -> None:
        """Verify full path: Query -> Search -> Filter -> Synthesize -> Finish."""
        # Step 1: Planner proposes SEARCH
        resp_1 = json.dumps({
            "action_rationale": "Search lectures for backpropagation.",
            "decision_summary": "Search backpropagation.",
            "action_type": "SEARCH",
            "tool_call": {
                "call_id": "call_1",
                "tool_name": "academic_search",
                "arguments": {"query": "backpropagation algorithm"},
            },
            "direct_answer": None,
            "refusal_reason": None,
        })
        # Step 2: Planner proposes FILTER
        resp_2 = json.dumps({
            "action_rationale": "Filter retrieved passages.",
            "decision_summary": "Filter candidates.",
            "action_type": "FILTER",
            "tool_call": {
                "call_id": "call_2",
                "tool_name": "evidence_filter",
                "arguments": {"candidates": []},
            },
            "direct_answer": None,
            "refusal_reason": None,
        })
        # Step 3: Planner proposes SYNTHESIZE
        resp_3 = json.dumps({
            "action_rationale": "Synthesize answer with citations.",
            "decision_summary": "Synthesize grounded answer.",
            "action_type": "SYNTHESIZE",
            "tool_call": {
                "call_id": "call_3",
                "tool_name": "academic_synthesizer",
                "arguments": {"query": "backpropagation algorithm"},
            },
            "direct_answer": None,
            "refusal_reason": None,
        })

        provider = MockLLMProvider([resp_1, resp_2, resp_3])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MockEvidenceFilterTool())
        registry.register(MockAcademicSynthesizerTool())

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace: AgentTrace = agent.run("How does backpropagation work in neural networks?")

        assert trace.terminal_state == AgentTerminalState.TERMINATED_SUCCESS.value
        assert trace.final_answer is not None
        assert "Lecture 04, p.12" in (trace.final_answer.resolved_answer_text or "")
        assert len(trace.steps) == 3
        assert trace.steps[0].tool_call.tool_name == "academic_search"
        assert trace.steps[1].tool_call.tool_name == "evidence_filter"
        assert trace.steps[2].tool_call.tool_name == "academic_synthesizer"
        assert len(trace.transitions) > 0

    def test_query_screening_out_of_domain(self) -> None:
        """Verify out-of-domain query is screened out immediately without tool execution."""
        provider = MockLLMProvider([])
        registry = ToolRegistry()
        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("Can you give me a recipe for chocolate cake?")

        assert trace.terminal_state == AgentTerminalState.TERMINATED_ABSTENTION.value
        assert "recipe" in (trace.refusal_explanation or "").lower() or "outside academic" in (trace.refusal_explanation or "").lower()
        assert len(trace.steps) == 0  # No tool steps executed
        assert provider.call_count == 0  # No LLM calls wasted

    def test_query_screening_underspecified(self) -> None:
        """Verify underspecified query triggers early clarification without tool calls."""
        provider = MockLLMProvider([])
        registry = ToolRegistry()
        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("what?")

        assert trace.terminal_state == AgentTerminalState.TERMINATED_ABSTENTION.value
        assert "underspecified" in (trace.refusal_explanation or "").lower()
        assert len(trace.steps) == 0

    def test_evidence_insufficiency_synthesis_abstention(self) -> None:
        """Verify when synthesizer reports refusal/insufficient evidence, agent terminates in abstention."""
        resp_1 = json.dumps({
            "action_rationale": "Search lectures.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {
                "call_id": "call_1",
                "tool_name": "academic_search",
                "arguments": {"query": "quantum gravity"},
            },
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize answer.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {
                "call_id": "call_2",
                "tool_name": "academic_synthesizer",
                "arguments": {"query": "quantum gravity"},
            },
        })

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        # Synthesizer configured with is_valid=False (insufficient evidence)
        registry.register(MockAcademicSynthesizerTool(is_valid=False))

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("Explain quantum gravity in Lecture 1.")

        assert trace.terminal_state == AgentTerminalState.TERMINATED_ABSTENTION.value
        assert "insufficient" in (trace.refusal_explanation or "").lower()


# ============================================================================
# Stage 6.4.8 — Adversarial, Failure & Concurrency Tests
# ============================================================================

class TestOrchestratorAdversarialAndConcurrency:
    """Test cancellation, budget exhaustion, policy enforcement, and multithreading."""

    def test_cooperative_cancellation_interruption(self) -> None:
        """Verify cooperative cancellation stops execution before subsequent steps."""
        token = CancellationToken()

        resp_1 = json.dumps({
            "action_rationale": "Search lectures.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {
                "call_id": "call_1",
                "tool_name": "academic_search",
                "arguments": {"query": "deep learning"},
            },
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize answer.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {
                "call_id": "call_2",
                "tool_name": "academic_synthesizer",
                "arguments": {"query": "deep learning"},
            },
        })

        # Cancel token after step 1
        class CancellingSearchTool(MockAcademicSearchTool):
            def execute(self, arguments: dict) -> ToolResult:
                token.cancel("User cancelled during search.")
                return super().execute(arguments)

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(CancellingSearchTool())
        registry.register(MockAcademicSynthesizerTool())

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("What is deep learning?", cancellation_token=token)

        assert trace.terminal_state == AgentTerminalState.TERMINATED_CANCELLED.value
        assert "User cancelled" in (trace.refusal_explanation or "")
        assert len(trace.steps) == 1  # Step 2 was never executed

    def test_iteration_budget_exhaustion(self) -> None:
        """Verify exceeding iteration ceiling terminates in TERMINATED_BUDGET_EXCEEDED."""
        # Planner proposes infinite SEARCH steps
        search_resp = json.dumps({
            "action_rationale": "Search lectures repeatedly.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {
                "call_id": "call_s",
                "tool_name": "academic_search",
                "arguments": {"query": "iteration test"},
            },
        })

        provider = MockLLMProvider([search_resp])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())

        planner = ReActStepPlanner(llm_provider=provider)
        config = AgentRunConfig(budget_config=BudgetConfig(max_iterations=2))
        agent = AcademicAgent(tool_registry=registry, planner=planner, default_config=config)

        trace = agent.run("Testing iteration budget bounds.")

        assert trace.terminal_state in (
            AgentTerminalState.TERMINATED_BUDGET_EXCEEDED.value,
            AgentTerminalState.TERMINATED_ABSTENTION.value,
        )
        assert trace.budget_consumed.iterations <= 3

    def test_consecutive_policy_rejection_ceiling(self) -> None:
        """Verify agent aborts when planner repeatedly proposes ungrounded direct answers."""
        # Planner proposes illegal FINISH with direct answer without prior synthesis
        illegal_resp = json.dumps({
            "action_rationale": "I know the answer without searching.",
            "decision_summary": "Direct ungrounded finish.",
            "action_type": "FINISH",
            "direct_answer": "Fabricated parametric answer.",
        })

        provider = MockLLMProvider([illegal_resp])
        registry = ToolRegistry()
        planner = ReActStepPlanner(llm_provider=provider)
        config = AgentRunConfig(max_consecutive_rejections=2)
        agent = AcademicAgent(tool_registry=registry, planner=planner, default_config=config)

        trace = agent.run("Explain eigenvalues and eigenvectors.")

        assert trace.terminal_state == AgentTerminalState.TERMINATED_ABSTENTION.value
        assert "consecutively rejected" in (trace.refusal_explanation or "")
        assert len(trace.steps) == 2
        assert all(step.authorized is False for step in trace.steps)

    def test_concurrent_agent_runs_isolation(self) -> None:
        """Verify multiple simultaneous agent executions in parallel threads do not leak state."""
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MockAcademicSynthesizerTool())

        def run_agent_task(idx: int) -> AgentTrace:
            query = f"Explain academic topic {idx} in detail."
            resp_1 = json.dumps({
                "action_rationale": f"Search topic {idx}.",
                "decision_summary": f"Search topic {idx}.",
                "action_type": "SEARCH",
                "tool_call": {
                    "call_id": f"call_s_{idx}",
                    "tool_name": "academic_search",
                    "arguments": {"query": f"topic_{idx}"},
                },
            })
            resp_2 = json.dumps({
                "action_rationale": f"Synthesize topic {idx}.",
                "decision_summary": f"Synthesize topic {idx}.",
                "action_type": "SYNTHESIZE",
                "tool_call": {
                    "call_id": f"call_syn_{idx}",
                    "tool_name": "academic_synthesizer",
                    "arguments": {"query": f"topic_{idx}"},
                },
            })
            provider = MockLLMProvider([resp_1, resp_2])
            planner = ReActStepPlanner(llm_provider=provider)
            agent = AcademicAgent(tool_registry=registry, planner=planner)
            return agent.run(query)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_agent_task, i) for i in range(5)]
            traces = [f.result() for f in futures]

        assert len(traces) == 5
        run_ids = {t.run_id for t in traces}
        assert len(run_ids) == 5  # Every run got a unique run_id

        for idx, t in enumerate(traces):
            assert f"topic {idx}" in t.query
            assert t.terminal_state == AgentTerminalState.TERMINATED_SUCCESS.value
            assert len(t.steps) == 2
            assert t.steps[0].tool_call.arguments["query"] == f"topic_{idx}"
