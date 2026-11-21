"""Unit tests for Stage 6.3.1 Planner Contracts and Structured Output Validation."""

import json
import pytest
from academic_agent.agent.models import ActionType, ProposedAction, ToolCall
from academic_agent.agent.parser import (
    StructuralValidationError,
    StructuredPlannerParser,
)


class TestPlannerContractsAndParser:
    """Test suite validating ActionType enumerations, structural contracts, and StructuredPlannerParser."""

    def test_action_type_enumeration(self):
        """ActionType contains exactly the 6 approved discrete agent actions."""
        values = {a.value for a in ActionType}
        assert values == {"SEARCH", "FILTER", "SYNTHESIZE", "CLARIFY", "ABSTAIN", "FINISH"}

    def test_parse_valid_search_action(self):
        """Valid SEARCH action with academic_search tool_call parses cleanly."""
        raw = """
        ```json
        {
            "action_rationale": "Student asks about AR(1) stationarity; searching Lecture 2.",
            "action_type": "SEARCH",
            "decision_summary": "Search Lecture 2 for AR(1) stationarity conditions",
            "tool_call": {
                "call_id": "call_search_1",
                "tool_name": "academic_search",
                "arguments": {
                    "query": "AR(1) stationarity condition",
                    "top_k": 3
                }
            }
        }
        ```
        """
        action = StructuredPlannerParser.parse(raw)
        assert isinstance(action, ProposedAction)
        assert action.action_type == ActionType.SEARCH
        assert action.action_rationale == "Student asks about AR(1) stationarity; searching Lecture 2."
        assert action.decision_summary == "Search Lecture 2 for AR(1) stationarity conditions"
        assert action.tool_call is not None
        assert action.tool_call.tool_name == "academic_search"
        assert action.tool_call.arguments["query"] == "AR(1) stationarity condition"
        assert action.tool_call.arguments["top_k"] == 3

    def test_parse_valid_filter_action(self):
        """Valid FILTER action with evidence_filter tool_call parses cleanly."""
        raw = json.dumps({
            "action_rationale": "Retrieved 5 candidates; filtering to top 2 by relevance.",
            "action_type": "FILTER",
            "decision_summary": "Filter candidates to top 2",
            "tool_call": {
                "call_id": "call_filter_1",
                "tool_name": "evidence_filter",
                "arguments": {
                    "candidates": [{"record_id": "r1", "score": 0.9}],
                    "max_candidates": 2
                }
            }
        })
        action = StructuredPlannerParser.parse(raw)
        assert action.action_type == ActionType.FILTER
        assert action.tool_call.tool_name == "evidence_filter"
        assert action.tool_call.arguments["max_candidates"] == 2

    def test_parse_valid_synthesize_action(self):
        """Valid SYNTHESIZE action with academic_synthesizer tool_call parses cleanly."""
        raw = """
        Here is the planned synthesis action:
        {
            "action_rationale": "Sufficient evidence acquired; synthesizing verified academic answer.",
            "action_type": "SYNTHESIZE",
            "decision_summary": "Synthesize answer using retrieved Lecture 2 passages",
            "tool_call": {
                "tool_name": "academic_synthesizer",
                "arguments": {
                    "query": "Explain AR(1) stationarity",
                    "evidence_candidates": [{"record_id": "r1", "text": "Stationarity requires |phi| < 1"}]
                }
            }
        }
        Awaiting dispatcher execution.
        """
        action = StructuredPlannerParser.parse(raw)
        assert action.action_type == ActionType.SYNTHESIZE
        assert action.tool_call.tool_name == "academic_synthesizer"

    def test_parse_valid_abstain_action(self):
        """Valid ABSTAIN action requires refusal_reason and forbids tool_call."""
        raw = json.dumps({
            "action_rationale": "Query is out of domain regarding cooking recipes.",
            "action_type": "ABSTAIN",
            "decision_summary": "Abstain due to out-of-domain query",
            "refusal_reason": "I can only answer questions based on the course materials."
        })
        action = StructuredPlannerParser.parse(raw)
        assert action.action_type == ActionType.ABSTAIN
        assert action.refusal_reason == "I can only answer questions based on the course materials."
        assert action.tool_call is None

    def test_parse_valid_clarify_action(self):
        """Valid CLARIFY action requires direct_answer containing clarification prompt."""
        raw = json.dumps({
            "action_rationale": "Query is underspecified ('what is it?'); asking student for course topic.",
            "action_type": "CLARIFY",
            "decision_summary": "Request student to specify course concept",
            "direct_answer": "Could you please specify which time series model or concept you are referring to?"
        })
        action = StructuredPlannerParser.parse(raw)
        assert action.action_type == ActionType.CLARIFY
        assert "specify which time series" in action.direct_answer
        assert action.tool_call is None

    def test_parse_valid_finish_action(self):
        """Valid FINISH action provides direct_answer delivery of validated answer."""
        raw = json.dumps({
            "action_rationale": "Verified synthesis complete; presenting response.",
            "action_type": "FINISH",
            "decision_summary": "Deliver verified final answer",
            "direct_answer": "An AR(1) process is stationary if |phi| < 1 [Lecture 2, p. 14]."
        })
        action = StructuredPlannerParser.parse(raw)
        assert action.action_type == ActionType.FINISH
        assert "stationary if |phi| < 1" in action.direct_answer
        assert action.tool_call is None

    def test_reject_empty_or_non_json_output(self):
        """Empty or non-JSON model output raises StructuralValidationError."""
        with pytest.raises(StructuralValidationError, match="empty or pure whitespace"):
            StructuredPlannerParser.parse("")

        with pytest.raises(StructuralValidationError, match="No JSON object detected"):
            StructuredPlannerParser.parse("I think we should search for Lecture 2.")

        with pytest.raises(StructuralValidationError, match="Failed to parse planner output as valid JSON"):
            StructuredPlannerParser.parse("{action_type: SEARCH, invalid_json}")

    def test_reject_missing_or_unknown_action_type(self):
        """Unknown or missing action_type raises StructuralValidationError."""
        raw_missing = json.dumps({
            "action_rationale": "Doing something",
            "decision_summary": "Doing something"
        })
        with pytest.raises(StructuralValidationError, match="Missing or invalid 'action_type'"):
            StructuredPlannerParser.parse(raw_missing)

        raw_unknown = json.dumps({
            "action_type": "EXECUTE_ARBITRARY_CODE",
            "action_rationale": "Executing code",
            "decision_summary": "Executing code"
        })
        with pytest.raises(StructuralValidationError, match="Unrecognized action_type 'EXECUTE_ARBITRARY_CODE'"):
            StructuredPlannerParser.parse(raw_unknown)

    def test_reject_missing_rationale_or_decision_summary(self):
        """Missing action_rationale or decision_summary raises StructuralValidationError."""
        raw_no_rationale = json.dumps({
            "action_type": "ABSTAIN",
            "decision_summary": "Summary",
            "refusal_reason": "Out of domain"
        })
        with pytest.raises(StructuralValidationError, match="must include a non-empty 'action_rationale'"):
            StructuredPlannerParser.parse(raw_no_rationale)

        raw_no_summary = json.dumps({
            "action_type": "ABSTAIN",
            "action_rationale": "Rationale",
            "refusal_reason": "Out of domain"
        })
        with pytest.raises(StructuralValidationError, match="must include a non-empty 'decision_summary'"):
            StructuredPlannerParser.parse(raw_no_summary)

    def test_reject_tool_action_without_tool_call(self):
        """Tool action (e.g. SEARCH) without tool_call object raises StructuralValidationError."""
        raw = json.dumps({
            "action_type": "SEARCH",
            "action_rationale": "Searching for stationarity",
            "decision_summary": "Search",
            "tool_call": None
        })
        with pytest.raises(StructuralValidationError, match="Action 'SEARCH' requires a 'tool_call' object"):
            StructuredPlannerParser.parse(raw)

    def test_reject_tool_action_with_mismatched_tool_name(self):
        """SEARCH action with tool_name='unrelated_tool' raises StructuralValidationError."""
        raw = json.dumps({
            "action_type": "SEARCH",
            "action_rationale": "Searching for stationarity",
            "decision_summary": "Search",
            "tool_call": {
                "tool_name": "evidence_filter",  # Mismatch: SEARCH must use academic_search
                "arguments": {}
            }
        })
        with pytest.raises(StructuralValidationError, match="Action 'SEARCH' requires tool_name='academic_search'"):
            StructuredPlannerParser.parse(raw)

    def test_reject_non_tool_action_with_dangling_tool_call(self):
        """Non-tool action (e.g. ABSTAIN or CLARIFY) with tool_call raises StructuralValidationError."""
        raw = json.dumps({
            "action_type": "ABSTAIN",
            "action_rationale": "Abstaining",
            "decision_summary": "Abstain",
            "refusal_reason": "No evidence",
            "tool_call": {"tool_name": "academic_search", "arguments": {}}
        })
        with pytest.raises(StructuralValidationError, match="must not include 'tool_call'"):
            StructuredPlannerParser.parse(raw)

    def test_reject_abstain_without_refusal_reason(self):
        """ABSTAIN action without refusal_reason raises StructuralValidationError."""
        raw = json.dumps({
            "action_type": "ABSTAIN",
            "action_rationale": "Abstaining",
            "decision_summary": "Abstain"
        })
        with pytest.raises(StructuralValidationError, match="must provide a non-empty 'refusal_reason'"):
            StructuredPlannerParser.parse(raw)

    def test_reject_clarify_without_direct_answer(self):
        """CLARIFY action without direct_answer raises StructuralValidationError."""
        raw = json.dumps({
            "action_type": "CLARIFY",
            "action_rationale": "Clarifying",
            "decision_summary": "Clarify"
        })
        with pytest.raises(StructuralValidationError, match="must provide 'direct_answer'"):
            StructuredPlannerParser.parse(raw)

    def test_reject_finish_without_direct_answer(self):
        """FINISH action without direct_answer raises StructuralValidationError."""
        raw = json.dumps({
            "action_type": "FINISH",
            "action_rationale": "Finishing",
            "decision_summary": "Finish"
        })
        with pytest.raises(StructuralValidationError, match="must provide 'direct_answer'"):
            StructuredPlannerParser.parse(raw)

    def test_parse_safe_helper(self):
        """parse_safe returns (action, None) on success and (None, error_str) on failure."""
        valid_raw = json.dumps({
            "action_type": "FINISH",
            "action_rationale": "Done",
            "decision_summary": "Done",
            "direct_answer": "Final response text"
        })
        action, err = StructuredPlannerParser.parse_safe(valid_raw)
        assert action is not None
        assert err is None
        assert action.action_type == ActionType.FINISH

        invalid_raw = "Not a json"
        action_bad, err_bad = StructuredPlannerParser.parse_safe(invalid_raw)
        assert action_bad is None
        assert err_bad is not None
        assert "No JSON object detected" in err_bad
