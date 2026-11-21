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
    def __init__(
        self,
        is_valid: bool = True,
        outcome: Optional[PolicyOutcome] = None,
        invalid_cited_ids: Optional[List[str]] = None,
    ) -> None:
        self.is_valid = is_valid
        self.outcome = outcome
        self.invalid_cited_ids = invalid_cited_ids or []

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
        effective_outcome = self.outcome or (PolicyOutcome.ANSWERED if self.is_valid else PolicyOutcome.COMPLIANT_ABSTENTION)
        valid_ids = ["cand_0"] if not self.invalid_cited_ids else []
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
                        is_valid=len(valid_ids) > 0,
                        resolution_label="Lecture 04, p.12",
                    )
                ],
                valid_cited_ids=valid_ids,
                invalid_cited_ids=self.invalid_cited_ids,
            ),
            policy_result=PolicyValidationResult(
                outcome=effective_outcome,
                is_compliant=(effective_outcome in {PolicyOutcome.ANSWERED, PolicyOutcome.PARTIAL_ANSWER, PolicyOutcome.COMPLIANT_ABSTENTION}),
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
                "is_valid_academic_answer": self.is_valid and (effective_outcome in {PolicyOutcome.ANSWERED, PolicyOutcome.PARTIAL_ANSWER}),
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


    def test_single_agent_instance_concurrent_requests(self) -> None:
        """Verify a single shared AcademicAgent instance safely handles parallel requests."""
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MockAcademicSynthesizerTool())

        # Thread-safe mock provider mapping query text to responses
        class DynamicQueryMockProvider:
            @property
            def model_name(self) -> str: return "shared-mock-provider"
            @property
            def context_window(self) -> int: return 4096
            @property
            def is_available(self) -> bool: return True
            def load(self) -> None: pass
            def release(self) -> None: pass

            def generate(self, messages: List[ChatMessage], config: Optional[LLMConfig] = None) -> LLMResponse:
                content_str = messages[-1].content
                # Parse query from prompt
                import re
                match = re.search(r'### USER QUERY:\s*"([^"]+)"', content_str)
                query_text = match.group(1) if match else "default"

                if "SEARCH" not in content_str:
                    # First turn: propose SEARCH
                    payload = {
                        "action_rationale": f"Search for {query_text}.",
                        "decision_summary": f"Search {query_text}.",
                        "action_type": "SEARCH",
                        "tool_call": {
                            "call_id": f"call_s_{abs(hash(query_text))}",
                            "tool_name": "academic_search",
                            "arguments": {"query": query_text},
                        },
                    }
                else:
                    # Second turn: propose SYNTHESIZE
                    payload = {
                        "action_rationale": f"Synthesize for {query_text}.",
                        "decision_summary": f"Synthesize {query_text}.",
                        "action_type": "SYNTHESIZE",
                        "tool_call": {
                            "call_id": f"call_syn_{abs(hash(query_text))}",
                            "tool_name": "academic_synthesizer",
                            "arguments": {"query": query_text},
                        },
                    }

                return LLMResponse(
                    content=json.dumps(payload),
                    role=MessageRole.ASSISTANT,
                    model_name=self.model_name,
                    latency_ms=20.0,
                    usage=LLMUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130),
                )

        provider = DynamicQueryMockProvider()
        planner = ReActStepPlanner(llm_provider=provider)
        # ONE shared AcademicAgent instance
        shared_agent = AcademicAgent(tool_registry=registry, planner=planner)

        def call_shared_agent(i: int) -> AgentTrace:
            return shared_agent.run(f"Concurrent academic question {i}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(call_shared_agent, i) for i in range(5)]
            traces = [f.result() for f in futures]

        assert len(traces) == 5
        run_ids = {t.run_id for t in traces}
        assert len(run_ids) == 5

        for idx, t in enumerate(traces):
            assert t.terminal_state == AgentTerminalState.TERMINATED_SUCCESS.value
            assert t.final_answer is not None
            assert len(t.steps) == 2

    def test_cancellation_before_run_starts(self) -> None:
        """Verify cancellation requested before run() begins aborts immediately without query screening."""
        token = CancellationToken()
        token.cancel("Cancelled before invocation.")

        provider = MockLLMProvider([])
        registry = ToolRegistry()
        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        # Even with an out-of-domain query, cancellation precedes query screening
        trace = agent.run("Can you give me a recipe for chocolate cake?", cancellation_token=token)

        assert trace.terminal_state == AgentTerminalState.TERMINATED_CANCELLED.value
        assert "Cancelled before invocation." in (trace.refusal_explanation or "")
        assert len(trace.steps) == 0

    def test_synthesis_missing_answer_fails_gracefully(self) -> None:
        """Verify that synthesis tool returning success without an AcademicAnswer instance transitions to ERROR."""
        class MalformedSynthesizerTool(MockAcademicSynthesizerTool):
            def execute(self, arguments: dict) -> ToolResult:
                # Returns success=True, but omits academic_answer!
                return ToolResult(
                    call_id="call_syn",
                    tool_name="academic_synthesizer",
                    success=True,
                    output={"answer_text": "text without object", "is_valid_academic_answer": True},
                )

        resp_1 = json.dumps({
            "action_rationale": "Search.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {"call_id": "c1", "tool_name": "academic_search", "arguments": {"query": "test"}},
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {"call_id": "c2", "tool_name": "academic_synthesizer", "arguments": {"query": "test"}},
        })

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MalformedSynthesizerTool())

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("Explain eigenvalues.")
        # Must NOT be TERMINATED_SUCCESS!
        assert trace.terminal_state == AgentTerminalState.TERMINATED_ERROR.value
        assert trace.final_answer is None
        assert "failed to provide a valid AcademicAnswer" in (trace.refusal_explanation or "")

    def test_synthesis_invalid_answer_type_fails(self) -> None:
        """Verify synthesis returning wrong type for academic_answer transitions to ERROR."""
        class WrongTypeSynthesizerTool(MockAcademicSynthesizerTool):
            def execute(self, arguments: dict) -> ToolResult:
                return ToolResult(
                    call_id="call_syn",
                    tool_name="academic_synthesizer",
                    success=True,
                    output={"academic_answer": "NotAnAcademicAnswerObject", "is_valid_academic_answer": True},
                )

        resp_1 = json.dumps({
            "action_rationale": "Search.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {"call_id": "c1", "tool_name": "academic_search", "arguments": {"query": "test"}},
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {"call_id": "c2", "tool_name": "academic_synthesizer", "arguments": {"query": "test"}},
        })

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(WrongTypeSynthesizerTool())

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("Explain eigenvalues.")
        assert trace.terminal_state == AgentTerminalState.TERMINATED_ERROR.value
        assert trace.final_answer is None

    def test_synthesis_missing_validity_defaults_false(self) -> None:
        """Verify synthesis returning answer without is_valid_academic_answer does not default to True."""
        class MissingValiditySynthesizerTool(MockAcademicSynthesizerTool):
            def execute(self, arguments: dict) -> ToolResult:
                res = super().execute(arguments)
                out = dict(res.output)
                out.pop("is_valid_academic_answer", None)
                return ToolResult(
                    call_id="call_syn",
                    tool_name="academic_synthesizer",
                    success=True,
                    output=out,
                )

        resp_1 = json.dumps({
            "action_rationale": "Search.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {"call_id": "c1", "tool_name": "academic_search", "arguments": {"query": "test"}},
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {"call_id": "c2", "tool_name": "academic_synthesizer", "arguments": {"query": "test"}},
        })

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MissingValiditySynthesizerTool())

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("Explain eigenvalues.")
        # Missing validity flag must default to False -> abstention or error, NEVER success
        assert trace.terminal_state != AgentTerminalState.TERMINATED_SUCCESS.value
        assert trace.terminal_state in (
            AgentTerminalState.TERMINATED_ABSTENTION.value,
            AgentTerminalState.TERMINATED_ERROR.value,
        )

    def test_deep_trace_immutability_multilevel(self) -> None:
        """Verify that completed traces cannot be mutated at nested levels."""
        resp_1 = json.dumps({
            "action_rationale": "Search.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {"call_id": "c1", "tool_name": "academic_search", "arguments": {"query": "immutable test"}},
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {"call_id": "c2", "tool_name": "academic_synthesizer", "arguments": {"query": "immutable test"}},
        })

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MockAcademicSynthesizerTool())

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("Explain immutability guarantees.")
        assert trace.terminal_state == AgentTerminalState.TERMINATED_SUCCESS.value

        # Level 1: Field assignment on frozen AgentTrace
        with pytest.raises(Exception):
            trace.terminal_state = "MUTATED"  # type: ignore

        # Level 2: Nested mutation on ToolCall.arguments
        step_0 = trace.steps[0]
        assert step_0.tool_call is not None
        with pytest.raises((TypeError, AttributeError)):
            step_0.tool_call.arguments["query"] = "MUTATED_QUERY"  # type: ignore

        # Level 3: Nested mutation on ToolResult.output dictionary
        assert step_0.tool_result is not None
        with pytest.raises((TypeError, AttributeError)):
            step_0.tool_result.output["count"] = 999  # type: ignore

        # Level 4: Nested mutation on nested candidate in output
        candidates = step_0.tool_result.output["candidates"]
        with pytest.raises((TypeError, AttributeError)):
            candidates[0]["score"] = 0.0  # type: ignore

        # Level 5: Nested mutation on frozen AcademicAnswer
        assert trace.final_answer is not None
        with pytest.raises((Exception, TypeError)):
            trace.final_answer.answer_text = "MUTATED_ANSWER"  # type: ignore

    def test_synthesis_policy_outcome_unsupported_answer_abstains(self) -> None:
        """Verify that synthesis with UNSUPPORTED_ANSWER transitions to TERMINATED_ABSTENTION."""
        resp_1 = json.dumps({
            "action_rationale": "Search.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {"call_id": "c1", "tool_name": "academic_search", "arguments": {"query": "test"}},
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {"call_id": "c2", "tool_name": "academic_synthesizer", "arguments": {"query": "test"}},
        })

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MockAcademicSynthesizerTool(is_valid=False, outcome=PolicyOutcome.UNSUPPORTED_ANSWER))

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("What is quantum backpropagation?")
        assert trace.terminal_state == AgentTerminalState.TERMINATED_ABSTENTION.value
        assert "unsupported" in (trace.refusal_explanation or "").lower()
        assert trace.transitions[-1].reason == "unsupported claims"
        assert trace.final_answer is not None

    def test_synthesis_policy_outcome_contradictory_refusal_errors(self) -> None:
        """Verify that synthesis with CONTRADICTORY_REFUSAL transitions to TERMINATED_ERROR."""
        resp_1 = json.dumps({
            "action_rationale": "Search.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {"call_id": "c1", "tool_name": "academic_search", "arguments": {"query": "test"}},
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {"call_id": "c2", "tool_name": "academic_synthesizer", "arguments": {"query": "test"}},
        })

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MockAcademicSynthesizerTool(is_valid=False, outcome=PolicyOutcome.CONTRADICTORY_REFUSAL))

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("Explain contradictory refusal.")
        assert trace.terminal_state == AgentTerminalState.TERMINATED_ERROR.value
        assert trace.transitions[-1].reason == "contradictory refusal"
        assert trace.final_answer is not None

    def test_synthesis_citation_integrity_failure_fabricated_ids(self) -> None:
        """Verify that synthesis with fabricated citation IDs transitions to TERMINATED_ERROR."""
        resp_1 = json.dumps({
            "action_rationale": "Search.",
            "decision_summary": "Search.",
            "action_type": "SEARCH",
            "tool_call": {"call_id": "c1", "tool_name": "academic_search", "arguments": {"query": "test"}},
        })
        resp_2 = json.dumps({
            "action_rationale": "Synthesize.",
            "decision_summary": "Synthesize.",
            "action_type": "SYNTHESIZE",
            "tool_call": {"call_id": "c2", "tool_name": "academic_synthesizer", "arguments": {"query": "test"}},
        })

        provider = MockLLMProvider([resp_1, resp_2])
        registry = ToolRegistry()
        registry.register(MockAcademicSearchTool())
        registry.register(MockAcademicSynthesizerTool(is_valid=True, invalid_cited_ids=["E99"]))

        planner = ReActStepPlanner(llm_provider=provider)
        agent = AcademicAgent(tool_registry=registry, planner=planner)

        trace = agent.run("Explain citation fabrication.")
        assert trace.terminal_state == AgentTerminalState.TERMINATED_ERROR.value
        assert trace.transitions[-1].reason == "citation integrity check failed"
        assert trace.final_answer is not None

    def test_planner_preflight_token_budget_clamping(self) -> None:
        """Verify that ReActStepPlanner clamps max_new_tokens based on remaining budget headroom."""
        captured_configs: List[LLMConfig] = []

        class RecordingLLMProvider:
            @property
            def model_name(self) -> str:
                return "recording-llm"
            @property
            def context_window(self) -> int:
                return 4096
            @property
            def is_available(self) -> bool:
                return True
            def generate(self, messages: Any, config: Optional[LLMConfig] = None) -> LLMResponse:
                if config:
                    captured_configs.append(config)
                return LLMResponse(
                    content=json.dumps({
                        "action_rationale": "Finish.",
                        "decision_summary": "Finish.",
                        "action_type": "FINISH",
                        "direct_answer": "Complete.",
                    }),
                    role=MessageRole.ASSISTANT,
                    model_name=self.model_name,
                    latency_ms=10.0,
                    usage=LLMUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
                )
            def load(self) -> None: pass
            def release(self) -> None: pass

        provider = RecordingLLMProvider()
        budget_cfg = BudgetConfig(max_total_tokens=500)
        tracker = ExecutionBudgetTracker(config=budget_cfg)
        # Consume 380 tokens so remaining tokens = 120
        tracker.record_tokens(prompt=300, completion=80, is_exact=True)

        planner = ReActStepPlanner(llm_provider=provider, max_tokens=1024)
        action = planner.plan_next_step(
            query="Test query",
            history=[],
            available_tools=[],
            budget_tracker=tracker,
        )

        assert action.action_type == ActionType.FINISH
        assert len(captured_configs) == 1
        # max_tokens was 1024, but headroom was 120 -> clamped to 120
        assert captured_configs[0].max_new_tokens == 120

        # Now test minimum clamping floor (16) when headroom is very low (e.g. 5 tokens left)
        captured_configs.clear()
        tracker2 = ExecutionBudgetTracker(config=BudgetConfig(max_total_tokens=100))
        tracker2.record_tokens(prompt=80, completion=15, is_exact=True)  # 5 tokens left

        action2 = planner.plan_next_step(
            query="Test query 2",
            history=[],
            available_tools=[],
            budget_tracker=tracker2,
        )
        assert len(captured_configs) == 1
        # remaining was 5, floor is 16 -> max(16, min(1024, 5)) == 16
        assert captured_configs[0].max_new_tokens == 16
        assert action2.action_type == ActionType.ABSTAIN
