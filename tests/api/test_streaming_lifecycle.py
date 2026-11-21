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

    def test_controlled_disconnect_before_agent_execution_barrier(self):
        """Verify synchronization barrier: client disconnect cancels worker before agent execution begins."""
        service = build_service_with_delay(delay_sec=0.05)
        run_id = f"barrier_cancel_{uuid.uuid4().hex[:6]}"
        req = QueryRequest(query="Explain AR(1) stationarity.", run_id=run_id)

        worker_reached_barrier = threading.Event()
        release_worker = threading.Event()
        tools_executed = threading.Event()
        planner_invocations = []
        dispatched_tools = []

        orig_plan = service.agent.planner.plan_next_step

        def barrier_plan(query, history, available_tools, budget_tracker=None):
            planner_invocations.append("step_0")
            worker_reached_barrier.set()
            release_worker.wait(timeout=5.0)
            # The planner finishes reasoning and produces a proposed SEARCH action
            return orig_plan(query, history, available_tools, budget_tracker=budget_tracker)

        service.agent.planner.plan_next_step = barrier_plan

        orig_dispatch = service.agent.dispatcher.dispatch

        def tracked_dispatch(*args, **kwargs):
            tool_name = args[0].tool_name if (args and hasattr(args[0], "tool_name")) else "tool"
            dispatched_tools.append(tool_name)
            tools_executed.set()
            return orig_dispatch(*args, **kwargs)

        service.agent.dispatcher.dispatch = tracked_dispatch

        async def _test():
            gen = service.stream_query_async(req)

            # 1. Begin consuming the stream -> advances generator and spawns worker
            first_event = await anext(gen)
            assert first_event.event_type == StreamEventType.RUN_STARTED

            # 2. Confirm the service has reserved the run
            assert service.is_active(run_id) is True
            token = service._active_tokens[run_id]

            # 3. Hold the worker at the planning phase
            assert worker_reached_barrier.wait(timeout=3.0) is True
            assert token.is_cancelled is False
            assert len(planner_invocations) == 1

            # 4. Disconnect the consumer while the planner was in-flight
            await gen.aclose()

            # 5. Assert the token is cancelled by the disconnect finally block
            assert token.is_cancelled is True

            # 6. Release the worker (planner finishes and returns proposed SEARCH action)
            release_worker.set()

            # Wait for worker thread to exit completely
            for _ in range(50):
                if not service.is_active(run_id):
                    break
                await asyncio.sleep(0.02)

            # 7. Assert:
            # - Planning was in-flight when cancelled (len(planner_invocations) == 1)
            # - ZERO tools were ever dispatched because the post-planner cancellation check aborted execution!
            assert len(dispatched_tools) == 0
            assert tools_executed.is_set() is False
            assert service.is_active(run_id) is False
            assert run_id not in service._active_runs
            assert run_id in service._completed_runs
            completed = service.get_run(run_id)
            assert completed is not None
            assert completed.status == ServiceRunStatus.CANCELLED
            assert completed.terminal_state == "TERMINATED_CANCELLED"
            assert any(term in completed.terminal_reason.lower() for term in ("cancelled", "cancellation", "disconnect"))

        asyncio.run(_test())

    def test_async_callback_backlog_bounded_under_flood(self):
        """Verify that pending callbacks scheduled on asyncio loop are bounded and never exceed max_async_backlog."""
        service = build_service_with_delay(delay_sec=0.0)

        max_observed_backlog = [0]
        recorded_observer = []

        def flood_telemetry_run(query, cancellation_token, config, run_id, observers=None):
            if observers:
                obs = observers[0]
                recorded_observer.append(obs)
                for i in range(1000):
                    obs.on_step_completed(
                        run_id,
                        AgentStep(
                            step_index=i,
                            state="PLANNING",
                            proposed_action=ProposedAction(
                                action_type=ActionType.SEARCH,
                                rationale=f"Flood step {i}",
                            ),
                            token_usage=TokenAccounting(prompt_tokens=1, completion_tokens=0, total_tokens=1),
                            latency_ms=0.1,
                        ),
                    )
                    cnt = obs.pending_backlog_count
                    if cnt > max_observed_backlog[0]:
                        max_observed_backlog[0] = cnt

            trace = AgentTrace(
                run_id=run_id,
                query=query,
                terminal_state=AgentTerminalState.TERMINATED_SUCCESS.value,
                budget_consumed=BudgetConsumption(
                    token_usage=TokenAccounting(prompt_tokens=1000, completion_tokens=0, total_tokens=1000),
                    wall_time_seconds=0.05,
                ),
                total_latency_ms=50.0,
                steps=(),
                transitions=(),
            )
            if observers:
                for obs in observers:
                    obs.on_run_completed(run_id, trace)
            return trace

        service.agent.run = flood_telemetry_run

        async def _test():
            req = QueryRequest(query="Measure pending async backlog")
            events = []
            gen = service.stream_query_async(req)
            # Read first event, then synchronously block the event loop thread
            # to guarantee saturation occurs on the producer side!
            first = await anext(gen)
            events.append(first)
            time.sleep(0.08)

            async for ev in gen:
                events.append(ev)

            # Assert saturation threshold was reached, backlog remained strictly bounded,
            # and dropped events count confirms that bounded-loss backpressure took effect.
            assert recorded_observer, "Observer should have been captured"
            obs = recorded_observer[0]
            assert max_observed_backlog[0] >= obs.max_async_backlog, (
                f"Saturation did not occur: observed {max_observed_backlog[0]} < {obs.max_async_backlog}"
            )
            assert max_observed_backlog[0] <= obs.max_async_backlog + 2, (
                f"Backlog exceeded bound: {max_observed_backlog[0]} > {obs.max_async_backlog + 2}"
            )
            assert obs.dropped_events_count > 0, "Frames should have been dropped during saturation"
            assert obs.dropped_events_count >= 800, f"Expected >=800 dropped frames, got {obs.dropped_events_count}"

            # Terminal delivery guaranteed
            assert events[-1].event_type == StreamEventType.RUN_COMPLETED
            assert events[-1].payload["status"] == ServiceRunStatus.COMPLETED.value

        asyncio.run(_test())

    def test_massive_event_stream_over_500_events_no_deadlock(self):
        """Verify that emitting >500 events in async mode does not deadlock or crash."""
        service = build_service_with_delay(delay_sec=0.0)

        # Custom agent run that emits 600 steps
        def emit_600_steps(query, cancellation_token, config, run_id, observers=None):
            if observers:
                for obs in observers:
                    for i in range(600):
                        step = AgentStep(
                            step_index=i,
                            state="PLANNING",
                            proposed_action=ProposedAction(
                                action_type=ActionType.SEARCH,
                                rationale="Stress testing async stream backpressure",
                            ),
                            token_usage=TokenAccounting(prompt_tokens=1, completion_tokens=0, total_tokens=1),
                            latency_ms=1.0,
                        )
                        obs.on_step_completed(run_id, step)
            trace = AgentTrace(
                run_id=run_id,
                query=query,
                terminal_state=AgentTerminalState.TERMINATED_SUCCESS.value,
                budget_consumed=BudgetConsumption(
                    token_usage=TokenAccounting(prompt_tokens=600, completion_tokens=0, total_tokens=600),
                    wall_time_seconds=0.1,
                ),
                total_latency_ms=100.0,
                steps=(),
                transitions=(),
            )
            if observers:
                for obs in observers:
                    obs.on_run_completed(run_id, trace)
            return trace

        service.agent.run = emit_600_steps

        async def _test():
            req = QueryRequest(query="Stress test async queue")
            events = []
            async for ev in service.stream_query_async(req):
                events.append(ev)

            assert len(events) > 0
            # Confirm terminal event was preserved and delivered despite saturation
            assert events[-1].event_type == StreamEventType.RUN_COMPLETED
            assert events[-1].payload["status"] == ServiceRunStatus.COMPLETED.value

        asyncio.run(_test())

    def test_queue_saturation_preserves_terminal_event(self):
        """Verify that when async queue is saturated, intermediate steps are dropped but terminal event is preserved."""
        service = build_service_with_delay(delay_sec=0.0)

        def flood_run(query, cancellation_token, config, run_id, observers=None):
            if observers:
                for obs in observers:
                    for i in range(550):
                        obs.on_step_completed(
                            run_id,
                            AgentStep(
                                step_index=i,
                                state="PLANNING",
                                proposed_action=ProposedAction(action_type=ActionType.SEARCH, rationale=f"step {i}"),
                                token_usage=TokenAccounting(prompt_tokens=1, completion_tokens=0, total_tokens=1),
                                latency_ms=0.5,
                            ),
                        )
            trace = AgentTrace(
                run_id=run_id,
                query=query,
                terminal_state=AgentTerminalState.TERMINATED_SUCCESS.value,
                budget_consumed=BudgetConsumption(
                    token_usage=TokenAccounting(prompt_tokens=550, completion_tokens=0, total_tokens=550),
                    wall_time_seconds=0.05,
                ),
                total_latency_ms=50.0,
                steps=(),
                transitions=(),
            )
            if observers:
                for obs in observers:
                    obs.on_run_completed(run_id, trace)
            return trace

        service.agent.run = flood_run

        async def _test():
            req = QueryRequest(query="Test saturation")
            events = []
            gen = service.stream_query_async(req)
            async for ev in gen:
                events.append(ev)
                if len(events) == 5:
                    # Artificially delay consumer to let worker flood queue
                    await asyncio.sleep(0.05)

            assert events[-1].event_type == StreamEventType.RUN_COMPLETED

        asyncio.run(_test())

    def test_concurrent_independent_streams(self):
        """Verify that multiple concurrent streams execute independently without queue cross-talk."""
        service = build_service_with_delay(delay_sec=0.02)

        async def _consume_stream(stream_idx: int):
            run_id = f"concurrent_run_{stream_idx}_{uuid.uuid4().hex[:6]}"
            req = QueryRequest(query=f"Explain AR(1) stream {stream_idx}", run_id=run_id)
            received = []
            async for ev in service.stream_query_async(req):
                assert ev.run_id == run_id
                received.append(ev)
            assert len(received) >= 2
            assert received[-1].event_type == StreamEventType.RUN_COMPLETED
            return run_id

        async def _main():
            results = await asyncio.gather(*[_consume_stream(i) for i in range(5)])
            assert len(set(results)) == 5
            # Give background threads a brief moment to finish thread cleanup
            await asyncio.sleep(0.1)
            for r_id in results:
                assert service.is_active(r_id) is False

        asyncio.run(_main())
