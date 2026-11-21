"""Structured planner output parsing and two-stage structural contract validation."""

import json
import re
from typing import Any, Dict, Optional, Tuple
from pydantic import ValidationError

from academic_agent.agent.models import ActionType, ProposedAction, ToolCall


class StructuralValidationError(Exception):
    """Raised when LLM output violates the strict planner JSON response contract."""

    def __init__(self, message: str, raw_output: str = "", error_details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.raw_output = raw_output
        self.error_details = error_details


class StructuredPlannerParser:
    """Parses raw LLM planner responses and strictly validates structural contracts.

    Guarantees:
    - Extracts JSON from raw text, code fences (```json ... ```), or mixed prose.
    - Rejects malformed JSON syntax with descriptive error diagnostics.
    - Validates presence of concise action_rationale and decision_summary (no internal chain dumps).
    - Enforces discrete ActionType enumeration.
    - Validates tool-action parity (e.g. SEARCH requires academic_search tool_call).
    - Validates terminal-action payloads (ABSTAIN requires refusal_reason, CLARIFY requires direct_answer).
    """

    ACTION_TOOL_MAPPING = {
        ActionType.SEARCH: "academic_search",
        ActionType.FILTER: "evidence_filter",
        ActionType.SYNTHESIZE: "academic_synthesizer",
    }

    @staticmethod
    def extract_json_payload(raw_text: str) -> str:
        """Extract the JSON substring from raw model output, stripping code fences or prose."""
        if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
            raise StructuralValidationError("Planner output is empty or pure whitespace.", raw_output=raw_text or "")

        cleaned = raw_text.strip()

        # 1. Check for markdown code fences: ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()

        # 2. Check for outermost balanced curly braces
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1].strip()

        raise StructuralValidationError(
            "No JSON object detected in planner output. Expected a JSON object with 'action_type' and 'action_rationale'.",
            raw_output=raw_text,
        )

    @classmethod
    def parse(cls, raw_output: str) -> ProposedAction:
        """Parse raw LLM output into a structurally verified ProposedAction.

        Raises:
            StructuralValidationError: If JSON is invalid, fields are missing, or invariants fail.
        """
        # 1. Extract JSON string
        json_str = cls.extract_json_payload(raw_output)

        # 2. Decode JSON
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as err:
            raise StructuralValidationError(
                f"Failed to parse planner output as valid JSON: {err}",
                raw_output=raw_output,
                error_details=str(err),
            ) from err

        if not isinstance(payload, dict):
            raise StructuralValidationError(
                f"Planner output must be a JSON object (dict), got: {type(payload).__name__}",
                raw_output=raw_output,
            )

        # 3. Validate action_type enumeration
        raw_act = payload.get("action_type")
        if not raw_act or not isinstance(raw_act, str):
            raise StructuralValidationError(
                f"Missing or invalid 'action_type' field in planner output: {raw_act!r}",
                raw_output=raw_output,
            )

        act_clean = raw_act.upper().strip()
        if act_clean not in ActionType.__members__:
            valid_types = [a.value for a in ActionType]
            raise StructuralValidationError(
                f"Unrecognized action_type '{raw_act}'. Must be one of: {valid_types}",
                raw_output=raw_output,
            )
        action_type = ActionType[act_clean]

        # 4. Validate concise rationale & decision summary
        rationale = payload.get("action_rationale") or payload.get("thought")
        if not rationale or not isinstance(rationale, str) or not rationale.strip():
            raise StructuralValidationError(
                "Planner output must include a non-empty 'action_rationale' explaining the selected action.",
                raw_output=raw_output,
            )

        decision_summary = payload.get("decision_summary")
        if not decision_summary or not isinstance(decision_summary, str) or not decision_summary.strip():
            raise StructuralValidationError(
                "Planner output must include a non-empty 'decision_summary' summarizing the decision.",
                raw_output=raw_output,
            )

        # 5. Validate Action-Specific Structural Contracts
        raw_tool_call = payload.get("tool_call")

        # A. Tool Actions: SEARCH, FILTER, SYNTHESIZE
        if action_type in cls.ACTION_TOOL_MAPPING:
            if not raw_tool_call or not isinstance(raw_tool_call, dict):
                raise StructuralValidationError(
                    f"Action '{action_type.value}' requires a 'tool_call' object, got: {raw_tool_call!r}",
                    raw_output=raw_output,
                )

            tool_name = raw_tool_call.get("tool_name")
            expected_tool = cls.ACTION_TOOL_MAPPING[action_type]
            if not tool_name or tool_name.lower().strip() != expected_tool:
                raise StructuralValidationError(
                    f"Action '{action_type.value}' requires tool_name='{expected_tool}', got '{tool_name}'",
                    raw_output=raw_output,
                )

            arguments = raw_tool_call.get("arguments")
            if arguments is None or not isinstance(arguments, dict):
                raise StructuralValidationError(
                    f"Tool call for '{tool_name}' must provide an 'arguments' dictionary.",
                    raw_output=raw_output,
                )

            # Reconstruct verified ToolCall
            call_id = raw_tool_call.get("call_id") or f"call_{action_type.value.lower()}"
            tool_call_obj = ToolCall(
                call_id=call_id,
                tool_name=expected_tool,
                arguments=arguments,
            )

            return ProposedAction(
                action_rationale=rationale.strip(),
                action_type=action_type,
                decision_summary=decision_summary.strip(),
                tool_call=tool_call_obj,
            )

        # B. Non-Tool Actions: ABSTAIN, CLARIFY, FINISH
        if raw_tool_call is not None:
            raise StructuralValidationError(
                f"Action '{action_type.value}' is a non-tool terminal/interaction action and must not include 'tool_call'.",
                raw_output=raw_output,
            )

        if action_type == ActionType.ABSTAIN:
            refusal_reason = payload.get("refusal_reason")
            if not refusal_reason or not isinstance(refusal_reason, str) or not refusal_reason.strip():
                raise StructuralValidationError(
                    "Action 'ABSTAIN' must provide a non-empty 'refusal_reason' explaining why evidence is insufficient.",
                    raw_output=raw_output,
                )
            return ProposedAction(
                action_rationale=rationale.strip(),
                action_type=action_type,
                decision_summary=decision_summary.strip(),
                refusal_reason=refusal_reason.strip(),
            )

        if action_type == ActionType.CLARIFY:
            clarify_msg = payload.get("direct_answer") or payload.get("refusal_reason")
            if not clarify_msg or not isinstance(clarify_msg, str) or not clarify_msg.strip():
                raise StructuralValidationError(
                    "Action 'CLARIFY' must provide 'direct_answer' containing the clarification question for the student.",
                    raw_output=raw_output,
                )
            return ProposedAction(
                action_rationale=rationale.strip(),
                action_type=action_type,
                decision_summary=decision_summary.strip(),
                direct_answer=clarify_msg.strip(),
            )

        if action_type == ActionType.FINISH:
            direct_ans = payload.get("direct_answer")
            if not direct_ans or not isinstance(direct_ans, str) or not direct_ans.strip():
                raise StructuralValidationError(
                    "Action 'FINISH' must provide 'direct_answer' containing the completed verified response.",
                    raw_output=raw_output,
                )
            return ProposedAction(
                action_rationale=rationale.strip(),
                action_type=action_type,
                decision_summary=decision_summary.strip(),
                direct_answer=direct_ans.strip(),
            )

        raise StructuralValidationError(
            f"Unhandled action_type '{action_type.value}' in parser.",
            raw_output=raw_output,
        )

    @classmethod
    def parse_safe(cls, raw_output: str) -> Tuple[Optional[ProposedAction], Optional[str]]:
        """Safe parsing helper returning (action, None) on success or (None, error_msg) on failure."""
        try:
            return cls.parse(raw_output), None
        except StructuralValidationError as e:
            return None, e.message
        except Exception as e:
            return None, f"Unexpected parsing failure: {str(e)}"
