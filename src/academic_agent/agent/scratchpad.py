"""Compact observation serializer and multi-turn scratchpad formatting for agent reasoning."""

import html
import json
import re
from typing import Any, Dict, List, Mapping, Optional

from academic_agent.agent.models import AgentStep, ToolResult, unfreeze_data


def escape_untrusted_content(text: str) -> str:
    """Neutralize potential XML tag breakout attacks in retrieved content.

    Ensures lecture documents containing malicious instructions or closing tags
    cannot escape the <untrusted_observation> wrapper.
    """
    if not text:
        return ""
    # Neutralize closing tags or nested tags that could confuse the parser
    safe = re.sub(r"<\s*/?\s*untrusted_observation\s*>", "[UNTRUSTED_TAG_DEFUSED]", text, flags=re.IGNORECASE)
    return safe


def serialize_tool_result(tool_result: ToolResult, max_snippet_length: int = 120) -> str:
    """Format a ToolResult into a compact, high-signal textual observation.

    Args:
        tool_result: The executed ToolResult to format.
        max_snippet_length: Character limit for text snippets.

    Returns:
        Compact serialized string representation.
    """
    if not tool_result.success:
        category = tool_result.error_category.value if tool_result.error_category else "EXECUTION_ERROR"
        err_msg = tool_result.error_message or "Unknown failure"
        return f"[FAILED - {category}]: {err_msg} (latency: {tool_result.latency_ms:.1f}ms)"

    output = tool_result.output
    tool_name = tool_result.tool_name.lower().strip()

    if isinstance(output, (dict, Mapping)):
        if tool_name == "academic_search":
            return _serialize_search_output(output, max_snippet_length)
        elif tool_name == "evidence_filter":
            return _serialize_filter_output(output, max_snippet_length)
        elif tool_name == "academic_synthesizer":
            return _serialize_synthesizer_output(output, max_snippet_length)

    # Generic dictionary or fallback representation
    if isinstance(output, (dict, list, Mapping, tuple)):
        dumped = json.dumps(unfreeze_data(output), default=str)
        if len(dumped) > 400:
            return dumped[:397] + "..."
        return dumped

    text_val = str(output) if output is not None else "None"
    if len(text_val) > 400:
        return text_val[:397] + "..."
    return text_val


def _clean_snippet(text: Optional[str], max_len: int) -> str:
    """Normalize whitespace and truncate snippet."""
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) > max_len:
        return normalized[: max_len - 3] + "..."
    return normalized


def _serialize_search_output(output: Dict[str, Any], max_snippet_len: int) -> str:
    candidates = output.get("candidates", [])
    count = output.get("count", len(candidates))
    lines = [f"Found {count} candidate passage(s):"]

    for idx, cand in enumerate(candidates[:6]):  # Compact top 6
        cid = cand.get("candidate_id") or cand.get("id") or f"cand_{idx}"
        meta = cand.get("metadata") or {}
        src = cand.get("source_file") or meta.get("source_file") or meta.get("lecture_id") or "lecture"
        page = cand.get("page") or meta.get("page") or meta.get("page_number") or "?"
        score = cand.get("score")
        score_str = f"score={score:.3f}" if isinstance(score, (int, float)) else "score=N/A"
        raw_text = cand.get("content") or cand.get("text") or ""
        snippet = _clean_snippet(raw_text, max_snippet_len)
        lines.append(f"  - [{cid}] {src} (p.{page}, {score_str}): \"{snippet}\"")

    if count > 6:
        lines.append(f"  ... ({count - 6} more candidates truncated)")

    return "\n".join(lines)


def _serialize_filter_output(output: Dict[str, Any], max_snippet_len: int) -> str:
    retained = output.get("retained_candidates", [])
    omitted = output.get("omitted_candidates", [])
    retained_count = output.get("retained_count", len(retained))
    omitted_count = output.get("omitted_count", len(omitted))

    lines = [f"Evidence Filter: {retained_count} retained, {omitted_count} omitted."]

    if retained:
        lines.append("Retained Evidence:")
        for idx, cand in enumerate(retained[:5]):
            cid = cand.get("candidate_id") or cand.get("id") or f"cand_{idx}"
            src = cand.get("source_file") or "lecture"
            page = cand.get("page") or "?"
            raw_text = cand.get("content") or cand.get("text") or ""
            snippet = _clean_snippet(raw_text, max_snippet_len)
            lines.append(f"  - [{cid}] {src} (p.{page}): \"{snippet}\"")

    if omitted_count > 0:
        # Group reasons
        reason_counts: Dict[str, int] = {}
        for item in output.get("omission_audit", omitted):
            reason = item.get("category") or item.get("reason") or "EXCLUDED"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        breakdown = ", ".join(f"{r}: {cnt}" for r, cnt in reason_counts.items())
        lines.append(f"Omission breakdown: {breakdown}")

    return "\n".join(lines)


def _serialize_synthesizer_output(output: Dict[str, Any], max_snippet_len: int) -> str:
    sufficiency = output.get("sufficiency", "UNKNOWN")
    policy = output.get("policy_assessment", "UNKNOWN")
    citation_count = output.get("citation_count", 0)
    answer = output.get("answer", "")
    preview = _clean_snippet(answer, max_snippet_len * 2)

    lines = [
        f"Grounded Synthesis Result [Sufficiency: {sufficiency}, Policy: {policy}, Citations: {citation_count}]:",
        f"Answer Preview: \"{preview}\"",
    ]
    return "\n".join(lines)


def wrap_untrusted_observation(call_id: str, tool_name: str, observation_text: str) -> str:
    """Enclose serialized tool observations in isolated untrusted XML tags."""
    safe_text = escape_untrusted_content(observation_text)
    return (
        f'<untrusted_observation id="{call_id}" tool="{tool_name}">\n'
        f"{safe_text}\n"
        f"</untrusted_observation>"
    )


def format_scratchpad(history: List[AgentStep]) -> str:
    """Format step-by-step agent trajectory into a clean multi-turn scratchpad.

    Args:
        history: Sequence of previous AgentStep records.

    Returns:
        Structured text representation for the LLM prompt.
    """
    if not history:
        return "(No prior actions executed yet. This is step 1.)"

    blocks: List[str] = []

    for step in history:
        lines: List[str] = [f"=== Step {step.step_index + 1} ==="]
        lines.append(f"State: {step.state}")

        if step.action_rationale:
            lines.append(f"Rationale: {step.action_rationale}")
        if step.decision_summary:
            lines.append(f"Decision: {step.decision_summary}")

        if step.proposed_action:
            action_type = step.proposed_action.action_type.value
            lines.append(f"Proposed Action: {action_type}")

            if step.proposed_action.direct_answer:
                lines.append(f"Direct Response: \"{_clean_snippet(step.proposed_action.direct_answer, 200)}\"")
            if step.proposed_action.refusal_reason:
                lines.append(f"Refusal Reason: \"{step.proposed_action.refusal_reason}\"")

        if step.tool_call:
            tc = step.tool_call
            args_str = json.dumps(unfreeze_data(tc.arguments), default=str)
            if len(args_str) > 150:
                args_str = args_str[:147] + "..."
            lines.append(f"Tool Call: {tc.tool_name}({args_str})")
            lines.append(f"Authorized: {step.authorized}")

        if not step.authorized and step.authorization_rejection_reason:
            lines.append(f"Authorization Rejection: {step.authorization_rejection_reason}")

        if step.tool_result:
            serialized_obs = serialize_tool_result(step.tool_result)
            wrapped = wrap_untrusted_observation(
                call_id=step.tool_result.call_id,
                tool_name=step.tool_result.tool_name,
                observation_text=serialized_obs,
            )
            lines.append("Observation:")
            lines.append(wrapped)

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
