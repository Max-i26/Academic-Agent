"""Authoritative deterministic decision policy for agent action authorization."""

from enum import Enum
import json
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field

from academic_agent.agent.models import (
    ActionType,
    AgentExecutionState,
    AgentState,
    AgentStep,
    AgentTerminalState,
    ProposedAction,
    VALID_STATE_TRANSITIONS,
)
from academic_agent.agent.protocols import ExecutionBudgetTrackerProtocol, ToolRegistryProtocol


class PolicyRejectionCategory(str, Enum):
    """Categorization of decision policy rejections."""

    ILLEGAL_TRANSITION = "ILLEGAL_TRANSITION"
    BYPASS_SYNTHESIS = "BYPASS_SYNTHESIS"
    SYNTHESIS_WITHOUT_EVIDENCE = "SYNTHESIS_WITHOUT_EVIDENCE"
    DUPLICATE_CALL = "DUPLICATE_CALL"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_CONTRACT_VIOLATION = "TOOL_CONTRACT_VIOLATION"
    REPEATED_FAILURE = "REPEATED_FAILURE"


class PolicyDecision(BaseModel):
    """Authoritative outcome of evaluating a proposed action against agent guardrails."""

    model_config = ConfigDict(frozen=True)

    authorized: bool = Field(description="True if the action is approved for execution.")
    action: ProposedAction = Field(description="Action to execute (original or coerced fallback).")
    rejection_category: Optional[PolicyRejectionCategory] = Field(
        default=None, description="Category of rejection if authorization was denied."
    )
    rejection_reason: Optional[str] = Field(
        default=None, description="Human-readable explanation of rejection or override."
    )
    suggested_action: Optional[ProposedAction] = Field(
        default=None, description="Guidance to help planner recover on next turn."
    )


class DecisionPolicy:
    """Deterministic policy engine enforcing architectural invariants before action dispatch.

    Invariants enforced:
    1. Budget Exhaustion: Overrides actions to graceful ABSTAIN/FINISH when resources run out.
    2. State Machine: Validates target state reachability under VALID_STATE_TRANSITIONS.
    3. Grounded Synthesis Boundary: Forbids FINISH with direct academic answers without Phase 5 synthesis.
    4. Evidence Prerequisite: Forbids SYNTHESIZE without prior evidence retrieval.
    5. Tool Verification: Confirms tool registration and tool/action parity.
    6. Duplicate Call Prevention: Intercepts duplicate searches or repeated failed invocations.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistryProtocol] = None,
        allow_clarification_without_search: bool = True,
    ) -> None:
        self.tool_registry = tool_registry
        self.allow_clarification_without_search = allow_clarification_without_search

    def evaluate(
        self,
        proposed_action: ProposedAction,
        current_state: AgentState,
        history: List[AgentStep],
        budget_tracker: Optional[ExecutionBudgetTrackerProtocol] = None,
    ) -> PolicyDecision:
        """Evaluate a proposed action and return an authoritative PolicyDecision.

        Args:
            proposed_action: The action proposed by the LLM planner.
            current_state: Current state of the AgentStateMachine.
            history: Complete step history of the current agent run.
            budget_tracker: Resource tracker to check exhaustion limits.

        Returns:
            PolicyDecision indicating authorization, rejection, or fallback coercion.
        """
        # 1. Budget Exhaustion Guardrail
        if budget_tracker and budget_tracker.is_exhausted():
            if proposed_action.action_type in (ActionType.ABSTAIN, ActionType.FINISH):
                return PolicyDecision(authorized=True, action=proposed_action)

            exhausted_resource = budget_tracker.get_exhausted_resource() or "resource limit"
            fallback_action = ProposedAction(
                action_rationale=f"Budget exhausted ({exhausted_resource}). Terminating execution.",
                decision_summary="Execution budget exhausted; forced graceful abstention.",
                action_type=ActionType.ABSTAIN,
                refusal_reason=f"Operation terminated: budget ceiling reached for {exhausted_resource}.",
            )
            return PolicyDecision(
                authorized=False,
                action=fallback_action,
                rejection_category=PolicyRejectionCategory.BUDGET_EXHAUSTED,
                rejection_reason=f"Execution budget for {exhausted_resource} is exhausted. Tool execution denied.",
                suggested_action=fallback_action,
            )

        # 2. State Machine Reachability
        target_state = self._map_action_to_state(proposed_action.action_type)
        if target_state is not None:
            valid_targets = VALID_STATE_TRANSITIONS.get(current_state, frozenset())
            # For PLANNING and ACTION_SELECTION, tools or terminal transitions are valid
            if target_state not in valid_targets and current_state != target_state:
                # Check intermediate transition: e.g. from PLANNING -> ACTION_SELECTION -> TOOL_INVOCATION
                can_reach = self._is_state_reachable(current_state, target_state)
                if not can_reach:
                    return PolicyDecision(
                        authorized=False,
                        action=proposed_action,
                        rejection_category=PolicyRejectionCategory.ILLEGAL_TRANSITION,
                        rejection_reason=(
                            f"Illegal transition: cannot move from {current_state.value} to {target_state.value}."
                        ),
                    )

        # 3. Grounded Synthesis Boundary: Disallow FINISH with unverified direct answers
        if proposed_action.action_type == ActionType.FINISH:
            synthesis_succeeded = any(
                step.tool_call
                and step.tool_call.tool_name == "academic_synthesizer"
                and step.tool_result
                and step.tool_result.success
                for step in history
            )
            if not synthesis_succeeded:
                # Check if this run is an answered clarification or refusal
                is_non_academic = any(
                    step.proposed_action and step.proposed_action.action_type in (ActionType.CLARIFY, ActionType.ABSTAIN)
                    for step in history
                )
                if not is_non_academic:
                    return PolicyDecision(
                        authorized=False,
                        action=proposed_action,
                        rejection_category=PolicyRejectionCategory.BYPASS_SYNTHESIS,
                        rejection_reason=(
                            "Direct academic answers without synthesizer execution are forbidden. "
                            "You must execute academic_synthesizer to produce a grounded answer before FINISH."
                        ),
                    )

        # 4. Evidence Prerequisite for Synthesis
        if proposed_action.action_type == ActionType.SYNTHESIZE:
            search_executed = any(
                step.tool_call
                and step.tool_call.tool_name == "academic_search"
                and step.tool_result
                and step.tool_result.success
                for step in history
            )
            if not search_executed:
                return PolicyDecision(
                    authorized=False,
                    action=proposed_action,
                    rejection_category=PolicyRejectionCategory.SYNTHESIS_WITHOUT_EVIDENCE,
                    rejection_reason=(
                        "Cannot invoke academic_synthesizer without prior evidence retrieval. "
                        "Execute academic_search first."
                    ),
                )

        # 5. Tool Verification & Contract Parity
        if proposed_action.action_type in (ActionType.SEARCH, ActionType.FILTER, ActionType.SYNTHESIZE):
            if not proposed_action.tool_call:
                return PolicyDecision(
                    authorized=False,
                    action=proposed_action,
                    rejection_category=PolicyRejectionCategory.TOOL_CONTRACT_VIOLATION,
                    rejection_reason=f"Action type {proposed_action.action_type.value} requires a populated tool_call.",
                )

            tool_name = proposed_action.tool_call.tool_name
            expected_tool = {
                ActionType.SEARCH: "academic_search",
                ActionType.FILTER: "evidence_filter",
                ActionType.SYNTHESIZE: "academic_synthesizer",
            }.get(proposed_action.action_type)

            if expected_tool and tool_name.lower() != expected_tool.lower():
                return PolicyDecision(
                    authorized=False,
                    action=proposed_action,
                    rejection_category=PolicyRejectionCategory.TOOL_CONTRACT_VIOLATION,
                    rejection_reason=(
                        f"Action type {proposed_action.action_type.value} must invoke '{expected_tool}', "
                        f"got '{tool_name}'."
                    ),
                )

            if self.tool_registry and not self.tool_registry.has(tool_name):
                return PolicyDecision(
                    authorized=False,
                    action=proposed_action,
                    rejection_category=PolicyRejectionCategory.TOOL_NOT_FOUND,
                    rejection_reason=f"Requested tool '{tool_name}' is not registered in ToolRegistry.",
                )
        else:
            # Non-tool actions must not carry a tool call
            if proposed_action.tool_call is not None:
                return PolicyDecision(
                    authorized=False,
                    action=proposed_action,
                    rejection_category=PolicyRejectionCategory.TOOL_CONTRACT_VIOLATION,
                    rejection_reason=f"Action type {proposed_action.action_type.value} forbids tool_call.",
                )

        # 6. Duplicate Call Prevention
        if proposed_action.tool_call:
            is_dup, dup_reason = self._detect_duplicate_tool_call(proposed_action.tool_call, history)
            if is_dup:
                return PolicyDecision(
                    authorized=False,
                    action=proposed_action,
                    rejection_category=PolicyRejectionCategory.DUPLICATE_CALL,
                    rejection_reason=dup_reason or "Duplicate tool call detected.",
                )

        # All checks passed: action is authorized
        return PolicyDecision(
            authorized=True,
            action=proposed_action,
        )

    def _map_action_to_state(self, action_type: ActionType) -> Optional[AgentState]:
        """Map high-level action type to target AgentState."""
        mapping = {
            ActionType.SEARCH: AgentExecutionState.TOOL_INVOCATION,
            ActionType.FILTER: AgentExecutionState.TOOL_INVOCATION,
            ActionType.SYNTHESIZE: AgentExecutionState.ANSWER_SYNTHESIS,
            ActionType.CLARIFY: AgentTerminalState.TERMINATED_SUCCESS,
            ActionType.ABSTAIN: AgentTerminalState.TERMINATED_ABSTENTION,
            ActionType.FINISH: AgentTerminalState.TERMINATED_SUCCESS,
        }
        return mapping.get(action_type)

    def _is_state_reachable(self, current: AgentState, target: AgentState) -> bool:
        """Check if target state is reachable within 2 transitions (e.g. PLANNING -> ACTION_SELECTION -> TOOL_INVOCATION)."""
        direct_targets = VALID_STATE_TRANSITIONS.get(current, frozenset())
        if target in direct_targets:
            return True
        for intermediate in direct_targets:
            if target in VALID_STATE_TRANSITIONS.get(intermediate, frozenset()):
                return True
        return False

    def _detect_duplicate_tool_call(
        self,
        tool_call: Any,
        history: List[AgentStep],
    ) -> tuple[bool, Optional[str]]:
        """Detect identical successful calls or repeated identical failures."""
        call_name = tool_call.tool_name.lower().strip()
        call_args_json = json.dumps(tool_call.arguments, sort_keys=True, default=str)

        for step in history:
            if step.tool_call:
                prev_name = step.tool_call.tool_name.lower().strip()
                prev_args_json = json.dumps(step.tool_call.arguments, sort_keys=True, default=str)

                if prev_name == call_name and prev_args_json == call_args_json:
                    if step.tool_result and step.tool_result.success:
                        return (
                            True,
                            f"Duplicate call: Tool '{call_name}' was already executed with identical arguments in step {step.step_index + 1}.",
                        )
                    if step.tool_result and not step.tool_result.success:
                        return (
                            True,
                            f"Repeated failure: Tool '{call_name}' already failed with identical arguments in step {step.step_index + 1}.",
                        )

        return (False, None)
