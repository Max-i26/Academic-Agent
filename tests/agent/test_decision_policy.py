"""Tests for the deterministic DecisionPolicy engine."""

import pytest

from academic_agent.agent.budget import ExecutionBudgetTracker
from academic_agent.agent.decision_policy import (
    DecisionPolicy,
    PolicyDecision,
    PolicyRejectionCategory,
)
from academic_agent.agent.models import (
    ActionType,
    AgentExecutionState,
    AgentStep,
    BudgetConfig,
    ProposedAction,
    ToolCall,
    ToolResult,
    ToolSideEffectPolicy,
)
from academic_agent.agent.registry import ToolRegistry
from academic_agent.agent.tools.search import AcademicSearchTool


@pytest.fixture
def empty_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def mock_registry() -> ToolRegistry:
    reg = ToolRegistry()
    policy = ToolSideEffectPolicy()

    # Register mock search tool
    class DummySearch:
        name = "academic_search"
        description = "search"
        side_effect_policy = policy
        def get_schema(self): return {"name": "academic_search", "type": "object", "properties": {}}
        def execute(self, args): return ToolResult(call_id="1", tool_name="academic_search", success=True)

    class DummyFilter:
        name = "evidence_filter"
        description = "filter"
        side_effect_policy = policy
        def get_schema(self): return {"name": "evidence_filter", "type": "object", "properties": {}}
        def execute(self, args): return ToolResult(call_id="2", tool_name="evidence_filter", success=True)

    class DummySynth:
        name = "academic_synthesizer"
        description = "synth"
        side_effect_policy = policy
        def get_schema(self): return {"name": "academic_synthesizer", "type": "object", "properties": {}}
        def execute(self, args): return ToolResult(call_id="3", tool_name="academic_synthesizer", success=True)

    reg.register(DummySearch())
    reg.register(DummyFilter())
    reg.register(DummySynth())
    return reg


class TestDecisionPolicy:
    """Test suite for DecisionPolicy architectural invariants."""

    def test_authorize_valid_search(self, mock_registry: ToolRegistry) -> None:
        """Verify standard valid search action is authorized."""
        policy = DecisionPolicy(tool_registry=mock_registry)
        action = ProposedAction(
            action_rationale="Need to search for lecture topics.",
            decision_summary="Initiate lecture search.",
            action_type=ActionType.SEARCH,
            tool_call=ToolCall(call_id="c_1", tool_name="academic_search", arguments={"query": "eigenvalues"}),
        )

        decision = policy.evaluate(
            proposed_action=action,
            current_state=AgentExecutionState.PLANNING,
            history=[],
        )
        assert decision.authorized is True
        assert decision.rejection_category is None

    def test_reject_bypass_synthesis(self, mock_registry: ToolRegistry) -> None:
        """Verify proposing FINISH with direct academic answer without synthesis is rejected."""
        policy = DecisionPolicy(tool_registry=mock_registry)
        action = ProposedAction(
            action_rationale="I know the answer directly.",
            decision_summary="Direct finish.",
            action_type=ActionType.FINISH,
            direct_answer="Eigenvalues are roots of the characteristic polynomial.",
        )

        decision = policy.evaluate(
            proposed_action=action,
            current_state=AgentExecutionState.PLANNING,
            history=[],
        )
        assert decision.authorized is False
        assert decision.rejection_category == PolicyRejectionCategory.BYPASS_SYNTHESIS
        assert "academic_synthesizer" in (decision.rejection_reason or "")

    def test_authorize_finish_after_synthesis(self, mock_registry: ToolRegistry) -> None:
        """Verify FINISH is authorized when synthesis succeeded in prior step."""
        policy = DecisionPolicy(tool_registry=mock_registry)
        synth_step = AgentStep(
            step_index=0,
            state=AgentExecutionState.ANSWER_SYNTHESIS.value,
            action_rationale="Synthesize answer.",
            decision_summary="Executed synthesis.",
            authorized=True,
            tool_call=ToolCall(call_id="c_syn", tool_name="academic_synthesizer", arguments={"query": "test"}),
            tool_result=ToolResult(
                call_id="c_syn",
                tool_name="academic_synthesizer",
                success=True,
                output={"answer": "Synthesized academic answer [1]."},
            ),
        )

        finish_action = ProposedAction(
            action_rationale="Synthesis complete; return final verified answer.",
            decision_summary="Finish task.",
            action_type=ActionType.FINISH,
            direct_answer="Synthesized academic answer [1].",
        )

        decision = policy.evaluate(
            proposed_action=finish_action,
            current_state=AgentExecutionState.PLANNING,
            history=[synth_step],
        )
        assert decision.authorized is True

    def test_reject_synthesis_without_evidence(self, mock_registry: ToolRegistry) -> None:
        """Verify calling academic_synthesizer without prior search is rejected."""
        policy = DecisionPolicy(tool_registry=mock_registry)
        synth_action = ProposedAction(
            action_rationale="Jump straight to synthesis.",
            decision_summary="Synthesize.",
            action_type=ActionType.SYNTHESIZE,
            tool_call=ToolCall(call_id="c_s", tool_name="academic_synthesizer", arguments={"query": "test"}),
        )

        decision = policy.evaluate(
            proposed_action=synth_action,
            current_state=AgentExecutionState.PLANNING,
            history=[],
        )
        assert decision.authorized is False
        assert decision.rejection_category == PolicyRejectionCategory.SYNTHESIS_WITHOUT_EVIDENCE

    def test_reject_tool_contract_mismatch(self, mock_registry: ToolRegistry) -> None:
        """Verify tool action with wrong tool name is rejected."""
        policy = DecisionPolicy(tool_registry=mock_registry)
        mismatch_action = ProposedAction(
            action_rationale="Searching using synthesis tool.",
            decision_summary="Search.",
            action_type=ActionType.SEARCH,
            tool_call=ToolCall(call_id="c_bad", tool_name="academic_synthesizer", arguments={}),
        )

        decision = policy.evaluate(
            proposed_action=mismatch_action,
            current_state=AgentExecutionState.PLANNING,
            history=[],
        )
        assert decision.authorized is False
        assert decision.rejection_category == PolicyRejectionCategory.TOOL_CONTRACT_VIOLATION

    def test_reject_unregistered_tool(self, empty_registry: ToolRegistry) -> None:
        """Verify requesting tool not in registry is rejected."""
        policy = DecisionPolicy(tool_registry=empty_registry)
        action = ProposedAction(
            action_rationale="Search.",
            decision_summary="Search.",
            action_type=ActionType.SEARCH,
            tool_call=ToolCall(call_id="c_1", tool_name="academic_search", arguments={"query": "test"}),
        )

        decision = policy.evaluate(
            proposed_action=action,
            current_state=AgentExecutionState.PLANNING,
            history=[],
        )
        assert decision.authorized is False
        assert decision.rejection_category == PolicyRejectionCategory.TOOL_NOT_FOUND

    def test_reject_tool_call_in_non_tool_action(self, mock_registry: ToolRegistry) -> None:
        """Verify CLARIFY or ABSTAIN carrying a tool_call is rejected."""
        policy = DecisionPolicy(tool_registry=mock_registry)
        invalid_clarify = ProposedAction(
            action_rationale="Clarify.",
            decision_summary="Clarify.",
            action_type=ActionType.CLARIFY,
            tool_call=ToolCall(call_id="c_1", tool_name="academic_search", arguments={}),
            direct_answer="Which week's lecture?",
        )

        decision = policy.evaluate(
            proposed_action=invalid_clarify,
            current_state=AgentExecutionState.PLANNING,
            history=[],
        )
        assert decision.authorized is False
        assert decision.rejection_category == PolicyRejectionCategory.TOOL_CONTRACT_VIOLATION

    def test_intercept_duplicate_search(self, mock_registry: ToolRegistry) -> None:
        """Verify identical search query is blocked."""
        policy = DecisionPolicy(tool_registry=mock_registry)
        prev_search = AgentStep(
            step_index=0,
            state=AgentExecutionState.PLANNING.value,
            action_rationale="Initial search.",
            decision_summary="Search.",
            authorized=True,
            tool_call=ToolCall(call_id="c_1", tool_name="academic_search", arguments={"query": "RNN architecture"}),
            tool_result=ToolResult(
                call_id="c_1",
                tool_name="academic_search",
                success=True,
                output={"candidates": [], "count": 0},
            ),
        )

        dup_action = ProposedAction(
            action_rationale="Search again with identical query.",
            decision_summary="Search again.",
            action_type=ActionType.SEARCH,
            tool_call=ToolCall(call_id="c_2", tool_name="academic_search", arguments={"query": "RNN architecture"}),
        )

        decision = policy.evaluate(
            proposed_action=dup_action,
            current_state=AgentExecutionState.PLANNING,
            history=[prev_search],
        )
        assert decision.authorized is False
        assert decision.rejection_category == PolicyRejectionCategory.DUPLICATE_CALL

    def test_budget_exhaustion_forces_abstain(self, mock_registry: ToolRegistry) -> None:
        """Verify exhausted budget overrides proposed tool action to ABSTAIN."""
        policy = DecisionPolicy(tool_registry=mock_registry)
        tracker = ExecutionBudgetTracker(BudgetConfig(max_iterations=1))
        tracker.record_iteration()  # Now exhausted

        proposed_search = ProposedAction(
            action_rationale="Search for more topics.",
            decision_summary="Search.",
            action_type=ActionType.SEARCH,
            tool_call=ToolCall(call_id="c_1", tool_name="academic_search", arguments={"query": "test"}),
        )

        decision = policy.evaluate(
            proposed_action=proposed_search,
            current_state=AgentExecutionState.PLANNING,
            history=[],
            budget_tracker=tracker,
        )
        assert decision.authorized is False
        assert decision.rejection_category == PolicyRejectionCategory.BUDGET_EXHAUSTED
        assert decision.action.action_type == ActionType.ABSTAIN
        assert "budget ceiling reached" in (decision.action.refusal_reason or "")
