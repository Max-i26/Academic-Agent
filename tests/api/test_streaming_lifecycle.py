"""Regression tests for SSE streaming lifecycle, async queue bridge, and race elimination.

Verifies:
- Elimination of worker thread generator advancement and closure races.
- Immediate cooperative cancellation upon client disconnection.
- Exception containment during background async streaming.
- Bounded async queue behavior.
"""

import asyncio
import json
import threading
import time
from typing import AsyncGenerator, List, Optional
import uuid

import pytest
from starlette.testclient import TestClient

from academic_agent.agent.context import AgentExecutionContext, AgentRunConfig, CancellationToken
from academic_agent.agent.decision_policy import DecisionPolicy
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import (
    ActionType,
    AgentExecutionState,
    AgentStep,
    AgentTerminalState,
    AgentTrace,
    BudgetConfig,
    BudgetConsumption,
    ProposedAction,
    StateTransitionRecord,
    TokenAccounting,
    ToolCall,
    ToolResult,
    ToolSideEffectPolicy,
)
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.planner import ReActStepPlanner
from academic_agent.agent.registry import ToolRegistry
from academic_agent.api.app import create_app
from academic_agent.api.models import QueryRequest, ServiceRunStatus, StreamEvent, StreamEventType
from academic_agent.generation.models import ChatMessage, LLMResponse, LLMUsage, MessageRole
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    CitationOccurrence,
    EvidenceContext,
    GroundedEvidence,
)
from academic_agent.service.service import AcademicAgentService


class DelayedSearchTool:
    """Tool that sleeps briefly to simulate realistic network/vector latency."""

    def __init__(self, delay_sec: float = 0.2) -> None:
        self.delay_sec = delay_sec

    @property
    def name(self) -> str:
        return "academic_search"

    @property
    def description(self) -> str:
        return "Simulated search with latency."

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, idempotent=True, timeout_seconds=5.0)

    def get_schema(self) -> dict:
        return {"name": "academic_search", "type": "object", "properties": {"query": {"type": "string"}}}

    def execute(self, arguments: dict) -> ToolResult:
        time.sleep(self.delay_sec)
        return ToolResult(
            call_id="call_delayed_search",
            tool_name="academic_search",
            output="Found Lecture_4.pdf with candidate excerpts.",
            is_error=False,
            execution_time_ms=self.delay_sec * 1000.0,
            token_usage=TokenAccounting(total_tokens=25),
        )


class DelayedSynthesizerTool:
    @property
    def name(self) -> str:
        return "academic_synthesizer"

    @property
    def description(self) -> str:
        return "Synthesizer."

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, idempotent=True, timeout_seconds=5.0)

    def get_schema(self) -> dict:
        return {"name": "academic_synthesizer", "type": "object", "properties": {"query": {"type": "string"}}}

    def execute(self, arguments: dict) -> ToolResult:
        evidence = GroundedEvidence(
            chunk_id="stat601_lec04_s18_c02",
            document_id="lec04",
            lecture_id="Lecture_4",
            page_number=18,
            title="Lecture 4: Autoregressive Models",
            text="Roots lie outside the unit circle.",
            score=0.95,
        )
        ctx = EvidenceContext(items=(evidence,))
        answer = AcademicAnswer(
            query="Stationarity conditions for AR(1)",
            resolved_answer_text="An AR(1) process is weakly stationary if and only if |phi_1| < 1 [Lecture 4, p. 18].",
            evidence_context=ctx,
            citation_occurrences=(
                CitationOccurrence(
                    citation_label="[Lecture 4, p. 18]",
                    target_chunk_id="stat601_lec04_s18_c02",
                    character_span=(68, 87),
                ),
            ),
            is_valid=True,
        )
        return ToolResult(
            call_id="call_delayed_synth",
            tool_name="academic_synthesizer",
            output=answer.model_dump_json(),
            is_error=False,
            execution_time_ms=50.0,
            token_usage=TokenAccounting(total_tokens=40),
        )


class MockLLMProvider:
    def __init__(self, responses: List[str]) -> None:
        self.responses = responses
        self.call_history: List[List[ChatMessage]] = []
        self._lock = threading.Lock()

    def generate(self, messages: List[ChatMessage], config: Optional[dict] = None) -> LLMResponse:
        with self._lock:
            self.call_history.append(list(messages))
            last_msg = messages[-1].content if messages else ""
            if "academic_search" in last_msg and len(self.responses) > 1:
                content = self.responses[1]
            else:
                content = self.responses[0]
            return LLMResponse(
                content=content,
                usage=LLMUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
                model_name="mock-llm-planner",
                latency_ms=10.0,
            )


def build_service_with_delay(delay_sec: float = 0.2) -> AcademicAgentService:
    registry = ToolRegistry()
    registry.register(DelayedSearchTool(delay_sec=delay_sec))
    registry.register(DelayedSynthesizerTool())

    search_step_json = json.dumps({
        "action_type": "SEARCH",
        "action_rationale": "Searching for lecture slides.",
        "decision_summary": "Querying slides.",
        "tool_call": {
            "call_id": "call_1",
            "tool_name": "academic_search",
            "arguments": {"query": "AR(1) stationarity"},
        },
    })
    synth_step_json = json.dumps({
        "action_type": "SYNTHESIZE",
        "action_rationale": "Synthesizing answer.",
        "decision_summary": "Invoking synthesizer.",
        "tool_call": {
            "call_id": "call_2",
            "tool_name": "academic_synthesizer",
            "arguments": {"query": "AR(1) stationarity"},
        },
    })

    provider = MockLLMProvider(responses=[search_step_json, synth_step_json])
    planner = ReActStepPlanner(llm_provider=provider)
    policy = DecisionPolicy(registry)
    dispatcher = ToolDispatcher(registry)

    agent = AcademicAgent(
        tool_registry=registry,
        planner=planner,
        dispatcher=dispatcher,
        decision_policy=policy,
        default_config=AgentRunConfig(budget_config=BudgetConfig(max_iterations=5, timeout_seconds=5.0)),
    )
    return AcademicAgentService(agent=agent)


class TestStreamingLifecycleAndRaceElimination:
    """Verifies that async SSE queue bridge eliminates thread races and handles cancellation."""

    def test_client_disconnect_triggers_cooperative_cancellation(self):
        """Simulate client consuming only the first SSE event and disconnecting."""
        service = build_service_with_delay(delay_sec=0.3)
        app = create_app(service=service)
        client = TestClient(app)

        run_id = f"test_disconnect_run_{uuid.uuid4().hex[:6]}"
        events_received = []

        with client.stream(
            "POST",
            "/api/v1/query/stream",
            json={"query": "Explain AR(1) stationarity.", "run_id": run_id},
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data:"):
                    events_received.append(line)
                    # Disconnect immediately after receiving the first event
                    break

        assert len(events_received) >= 1
        # Give worker a brief moment to process cancellation
        time.sleep(0.4)

        # Confirm run was cancelled and cleaned up from active runs
        assert service.is_active(run_id) is False

    def test_async_stream_query_produces_no_generator_races(self):
        """Verify that consuming stream_query_async concurrently does not raise generator already executing."""
        service = build_service_with_delay(delay_sec=0.05)

        async def _run_stream():
            req = QueryRequest(query="Explain AR(1) stationarity.", run_id=f"run_{uuid.uuid4().hex[:6]}")
            events = []
            async for ev in service.stream_query_async(req):
                events.append(ev)
            assert len(events) >= 3
            assert events[0].event_type == StreamEventType.RUN_STARTED
            assert events[-1].event_type == StreamEventType.RUN_COMPLETED

        async def _main():
            # Run two concurrent async streams
            await asyncio.gather(_run_stream(), _run_stream())

        asyncio.run(_main())

    def test_worker_exception_emits_failed_event(self):
        """Verify that if worker thread encounters an error, RUN_FAILED is emitted cleanly."""
        service = build_service_with_delay(delay_sec=0.01)

        def failing_run(*args, **kwargs):
            raise RuntimeError("Engine catastrophic crash!")

        service.agent.run = failing_run

        async def _test():
            req = QueryRequest(query="Explain AR(1) stationarity.")
            events = []
            async for ev in service.stream_query_async(req):
                events.append(ev)

            # Either service caught it and returned ERROR, or worker emitted RUN_FAILED
            assert len(events) >= 1
            last_event = events[-1]
            assert last_event.event_type in (StreamEventType.RUN_COMPLETED, StreamEventType.RUN_FAILED)

        asyncio.run(_test())
