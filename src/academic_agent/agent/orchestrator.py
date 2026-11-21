"""AcademicAgent orchestrator implementing the full bounded autonomous loop."""

import logging
import time
from typing import List, Optional

from academic_agent.agent.context import (
    AgentExecutionContext,
    AgentRunConfig,
    CancellationToken,
)
from academic_agent.agent.decision_policy import DecisionPolicy, PolicyDecision
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import (
    ActionType,
    AgentExecutionState,
    AgentStep,
    AgentTerminalState,
    AgentTrace,
    ProposedAction,
    ToolCall,
    ToolResult,
)
from academic_agent.agent.planner import ReActStepPlanner
from academic_agent.agent.protocols import (
    AgentLifecycleObserverProtocol,
    AgentProtocol,
    StepPlannerProtocol,
    ToolDispatcherProtocol,
    ToolRegistryProtocol,
)
from academic_agent.agent.tools.policy import ClarificationPolicy, QueryAssessmentCategory
from academic_agent.generation.synthesis.models import AcademicAnswer

logger = logging.getLogger(__name__)


class AcademicAgent(AgentProtocol):
    """Authoritative autonomous academic assistant orchestrating bounded reasoning and tool execution.

    Coordinates:
    - Pre-retrieval query screening (ClarificationPolicy).
    - Step-by-step reasoning proposals (StepPlannerProtocol / ReActStepPlanner).
    - Deterministic action authorization (DecisionPolicy).
    - Isolated tool dispatching and bounded execution (ToolDispatcher).
    - Verified grounded answer synthesis (Phase 5).
    - Deeply immutable audit trace generation (AgentTrace).
    """

    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        planner: StepPlannerProtocol,
        dispatcher: Optional[ToolDispatcherProtocol] = None,
        decision_policy: Optional[DecisionPolicy] = None,
        default_config: Optional[AgentRunConfig] = None,
        observers: Optional[List[AgentLifecycleObserverProtocol]] = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.planner = planner
        self.dispatcher = dispatcher or ToolDispatcher(tool_registry)
        self.decision_policy = decision_policy or DecisionPolicy(tool_registry)
        self.default_config = default_config or AgentRunConfig()
        self.observers = observers or []

    def run(
        self,
        query: str,
        cancellation_token: Optional[CancellationToken] = None,
        config: Optional[AgentRunConfig] = None,
    ) -> AgentTrace:
        """Execute bounded academic research and synthesis for the user query.

        Args:
            query: Student's academic question.
            cancellation_token: Optional token for cooperative interruption.
            config: Optional per-run configuration overrides.

        Returns:
            Immutable AgentTrace documenting the entire execution lifecycle.
        """
        run_config = config or self.default_config
        context = AgentExecutionContext(
            query=query,
            config=run_config,
            cancellation_token=cancellation_token,
        )

        self._notify_run_started(context.run_id, query)

        try:
            self._execute_lifecycle(context)
        except Exception as exc:
            logger.exception("Unhandled error in agent run %s: %s", context.run_id, exc)
            self._handle_unexpected_failure(context, exc)

        trace = context.build_trace()
        self._notify_run_completed(context.run_id, trace)
        return trace

    def _execute_lifecycle(self, context: AgentExecutionContext) -> None:
        """Drive the state machine and loop until a terminal state is reached."""
        # 1. Pre-retrieval query screening
        if context.config.enable_query_screening:
            if self._screen_query(context):
                return

        # Advance from INITIALIZED to PLANNING
        context.transition_to(AgentExecutionState.PLANNING, reason="Query validated; initiating planning cycle")

        step_counter = 0

        # Main Reasoning and Execution Loop
        while not context.is_terminated:
            # A. Check cooperative cancellation
            if context.check_cancellation():
                context.transition_to(
                    AgentTerminalState.TERMINATED_CANCELLED,
                    reason=context.refusal_explanation or "Cooperative cancellation requested.",
                )
                break

            # B. Check global budget and timeout bounds
            if context.budget_tracker.is_exhausted():
                exhausted = context.budget_tracker.get_exhausted_resource() or "budget ceiling"
                context.refusal_explanation = f"Execution terminated: {exhausted} exhausted."
                context.transition_to(
                    AgentTerminalState.TERMINATED_BUDGET_EXCEEDED,
                    reason=f"Resource {exhausted} exhausted before step {step_counter + 1}.",
                )
                break

            try:
                context.budget_tracker.check_time()
            except Exception as timeout_exc:
                context.refusal_explanation = str(timeout_exc)
                context.transition_to(
                    AgentTerminalState.TERMINATED_BUDGET_EXCEEDED,
                    reason="Wall-clock execution timeout ceiling exceeded.",
                )
                break

            # C. Execute single reasoning iteration
            self._execute_iteration(context, step_counter)
            step_counter += 1

    def _screen_query(self, context: AgentExecutionContext) -> bool:
        """Screen query with ClarificationPolicy. Returns True if screened to early termination."""
        assessment = ClarificationPolicy.assess_query(context.query)

        if assessment.category == QueryAssessmentCategory.OUT_OF_DOMAIN:
            context.transition_to(
                AgentExecutionState.PLANNING,
                reason="Query screened as out-of-domain",
            )
            context.refusal_explanation = assessment.rationale
            context.transition_to(
                AgentTerminalState.TERMINATED_ABSTENTION,
                reason=f"Out of domain query: {assessment.rationale}",
            )
            return True

        if assessment.category == QueryAssessmentCategory.UNDERSPECIFIED:
            context.transition_to(
                AgentExecutionState.PLANNING,
                reason="Query screened as underspecified",
            )
            clarification_question = (
                "Your question is underspecified or ambiguous. "
                "Please specify the topic, lecture, or specific academic concept you would like answered."
            )
            context.refusal_explanation = clarification_question
            context.transition_to(
                AgentTerminalState.TERMINATED_ABSTENTION,
                reason="Underspecified query requiring student clarification",
            )
            return True

        if assessment.category == QueryAssessmentCategory.MALFORMED:
            context.transition_to(
                AgentExecutionState.PLANNING,
                reason="Query screened as malformed",
            )
            context.refusal_explanation = assessment.rationale
            context.transition_to(
                AgentTerminalState.TERMINATED_ABSTENTION,
                reason="Malformed or empty query",
            )
            return True

        return False

    def _execute_iteration(self, context: AgentExecutionContext, step_index: int) -> None:
        """Execute one complete planning, authorization, execution, and observation cycle."""
        step_start_time = time.perf_counter()

        # Record iteration in budget tracker
        try:
            context.budget_tracker.record_iteration()
        except Exception as iter_exc:
            context.refusal_explanation = str(iter_exc)
            context.transition_to(
                AgentTerminalState.TERMINATED_BUDGET_EXCEEDED,
                reason="Maximum planning iterations ceiling reached.",
            )
            return

        # 1. Planning phase (executed in PLANNING state)
        available_tools = self.tool_registry.list_tools()
        proposed_action: ProposedAction = self.planner.plan_next_step(
            query=context.query,
            history=context.steps,
            available_tools=available_tools,
        )

        # 2. Authorization phase (evaluated while in PLANNING)
        decision: PolicyDecision = self.decision_policy.evaluate(
            proposed_action=proposed_action,
            current_state=context.current_state,
            history=context.steps,
            budget_tracker=context.budget_tracker,
        )

        # Handle Policy Rejection
        if not decision.authorized:
            context.consecutive_rejections += 1
            step_latency_ms = (time.perf_counter() - step_start_time) * 1000.0

            rejected_step = AgentStep(
                step_index=step_index,
                state=AgentExecutionState.PLANNING.value,
                action_rationale=proposed_action.action_rationale,
                decision_summary=proposed_action.decision_summary or "Action rejected by policy.",
                proposed_action=proposed_action,
                authorized=False,
                authorization_rejection_reason=decision.rejection_reason,
                latency_ms=round(step_latency_ms, 2),
            )
            context.add_step(rejected_step)
            self._notify_step_completed(context.run_id, rejected_step)

            # Check consecutive rejections ceiling
            if context.consecutive_rejections >= context.config.max_consecutive_rejections:
                context.refusal_explanation = (
                    f"Execution aborted: planner generated {context.consecutive_rejections} "
                    f"consecutively rejected actions ({decision.rejection_reason})."
                )
                context.transition_to(
                    AgentTerminalState.TERMINATED_ABSTENTION,
                    reason="Max consecutive policy rejections threshold breached.",
                )
                return

            # If the decision coerced an ABSTAIN action (e.g. budget exhaustion)
            if decision.action.action_type == ActionType.ABSTAIN:
                context.refusal_explanation = decision.action.refusal_reason or decision.rejection_reason
                context.transition_to(
                    AgentTerminalState.TERMINATED_ABSTENTION,
                    reason=decision.rejection_reason or "Policy forced abstention",
                )
                return

            # Action rejected; remains in PLANNING to re-plan with updated scratchpad
            return

        # Action is Authorized -> Transition to ACTION_SELECTION
        context.consecutive_rejections = 0
        action = decision.action

        context.transition_to(
            AgentExecutionState.ACTION_SELECTION,
            reason=f"Action authorized: {action.action_type.value}",
        )

        # Route Non-Tool Abstention
        if action.action_type == ActionType.ABSTAIN:
            step_latency_ms = (time.perf_counter() - step_start_time) * 1000.0
            abstain_step = AgentStep(
                step_index=step_index,
                state=AgentExecutionState.ACTION_SELECTION.value,
                action_rationale=action.action_rationale,
                decision_summary=action.decision_summary or "Agent decided to abstain.",
                proposed_action=action,
                authorized=True,
                latency_ms=round(step_latency_ms, 2),
            )
            context.add_step(abstain_step)
            self._notify_step_completed(context.run_id, abstain_step)
            context.refusal_explanation = action.refusal_reason or "Agent abstained from answering."
            context.transition_to(
                AgentTerminalState.TERMINATED_ABSTENTION,
                reason=action.refusal_reason or "Intentional agent abstention.",
            )
            return

        # Route Tool Actions (SEARCH, FILTER, SYNTHESIZE)
        tool_call = action.tool_call
        assert tool_call is not None, "Authorized tool action must possess a valid tool_call."

        if action.action_type == ActionType.SYNTHESIZE:
            self._execute_synthesis_step(context, action, tool_call, step_index, step_start_time)
        else:
            self._execute_tool_step(context, action, tool_call, step_index, step_start_time)

    def _execute_tool_step(
        self,
        context: AgentExecutionContext,
        action: ProposedAction,
        tool_call: ToolCall,
        step_index: int,
        step_start_time: float,
    ) -> None:
        """Execute intermediate tool calls (SEARCH, FILTER) with isolated dispatch and observation processing."""
        context.transition_to(
            AgentExecutionState.TOOL_INVOCATION,
            reason=f"Dispatching tool {tool_call.tool_name}",
        )

        tool_result = self.dispatcher.dispatch(tool_call, budget_tracker=context.budget_tracker)

        context.transition_to(
            AgentExecutionState.OBSERVATION_PROCESSING,
            reason=f"Processing observation from {tool_call.tool_name}",
        )

        step_latency_ms = (time.perf_counter() - step_start_time) * 1000.0
        step = AgentStep(
            step_index=step_index,
            state=AgentExecutionState.TOOL_INVOCATION.value,
            action_rationale=action.action_rationale,
            decision_summary=action.decision_summary or f"Executed {tool_call.tool_name}.",
            proposed_action=action,
            authorized=True,
            tool_call=tool_call,
            tool_result=tool_result,
            latency_ms=round(step_latency_ms, 2),
        )
        context.add_step(step)
        self._notify_step_completed(context.run_id, step)

        # Transition back to PLANNING for the next turn
        context.transition_to(
            AgentExecutionState.PLANNING,
            reason="Observation incorporated into memory; re-entering planning",
        )

    def _execute_synthesis_step(
        self,
        context: AgentExecutionContext,
        action: ProposedAction,
        tool_call: ToolCall,
        step_index: int,
        step_start_time: float,
    ) -> None:
        """Execute grounded answer synthesis step via AcademicSynthesizerTool."""
        context.transition_to(
            AgentExecutionState.ANSWER_SYNTHESIS,
            reason="Invoking Phase 5 grounded synthesizer",
        )

        tool_result = self.dispatcher.dispatch(tool_call, budget_tracker=context.budget_tracker)
        step_latency_ms = (time.perf_counter() - step_start_time) * 1000.0

        if tool_result.success and isinstance(tool_result.output, dict):
            # Extract AcademicAnswer object if attached
            academic_ans = tool_result.output.get("academic_answer")
            context.final_answer = academic_ans

            is_valid = tool_result.output.get("is_valid_academic_answer", True)

            synth_step = AgentStep(
                step_index=step_index,
                state=AgentExecutionState.ANSWER_SYNTHESIS.value,
                action_rationale=action.action_rationale,
                decision_summary="Completed grounded academic synthesis.",
                proposed_action=action,
                authorized=True,
                tool_call=tool_call,
                tool_result=tool_result,
                latency_ms=round(step_latency_ms, 2),
            )
            context.add_step(synth_step)
            self._notify_step_completed(context.run_id, synth_step)

            if is_valid:
                context.transition_to(
                    AgentTerminalState.TERMINATED_SUCCESS,
                    reason="Grounded answer synthesized with valid citations.",
                )
            else:
                context.refusal_explanation = "Retrieved evidence was insufficient for grounded synthesis."
                context.transition_to(
                    AgentTerminalState.TERMINATED_ABSTENTION,
                    reason="Synthesis produced refusal or partial answer dominated by evidence gaps.",
                )
        else:
            # Synthesis tool crashed or timed out
            err_msg = tool_result.error_message or "Synthesis execution failed."
            context.refusal_explanation = err_msg

            failed_step = AgentStep(
                step_index=step_index,
                state=AgentExecutionState.ANSWER_SYNTHESIS.value,
                action_rationale=action.action_rationale,
                decision_summary="Synthesizer execution failed.",
                proposed_action=action,
                authorized=True,
                tool_call=tool_call,
                tool_result=tool_result,
                latency_ms=round(step_latency_ms, 2),
            )
            context.add_step(failed_step)
            self._notify_step_completed(context.run_id, failed_step)

            context.transition_to(
                AgentTerminalState.TERMINATED_ERROR,
                reason=f"Synthesizer failure: {err_msg}",
            )

    def _handle_unexpected_failure(self, context: AgentExecutionContext, exc: Exception) -> None:
        """Safely transition to TERMINATED_ERROR on unexpected runtime crash."""
        if not context.is_terminated:
            context.refusal_explanation = f"Unexpected agent exception: {str(exc)}"
            try:
                context.transition_to(
                    AgentTerminalState.TERMINATED_ERROR,
                    reason=f"Unhandled runtime exception: {str(exc)}",
                )
            except Exception:
                pass

    def _notify_run_started(self, run_id: str, query: str) -> None:
        for obs in self.observers:
            try:
                obs.on_run_started(run_id, query)
            except Exception as obs_exc:
                logger.warning("Observer on_run_started raised: %s", obs_exc)

    def _notify_step_completed(self, run_id: str, step: AgentStep) -> None:
        for obs in self.observers:
            try:
                obs.on_step_completed(run_id, step)
            except Exception as obs_exc:
                logger.warning("Observer on_step_completed raised: %s", obs_exc)

    def _notify_run_completed(self, run_id: str, trace: AgentTrace) -> None:
        for obs in self.observers:
            try:
                obs.on_run_completed(run_id, trace)
            except Exception as obs_exc:
                logger.warning("Observer on_run_completed raised: %s", obs_exc)
