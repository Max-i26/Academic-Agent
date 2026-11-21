"""ReAct Step Planner implementing StepPlannerProtocol with structured parsing and graceful repair."""

import logging
from typing import Any, Dict, List, Optional

from academic_agent.agent.models import (
    ActionType,
    AgentStep,
    ProposedAction,
    TokenAccounting,
)
from academic_agent.agent.parser import StructuralValidationError, StructuredPlannerParser
from academic_agent.agent.prompts import build_planner_prompt, build_repair_prompt
from academic_agent.agent.protocols import (
    ExecutionBudgetTrackerProtocol,
    StepPlannerProtocol,
    ToolProtocol,
)
from academic_agent.agent.scratchpad import format_scratchpad
from academic_agent.generation.models import LLMConfig
from academic_agent.generation.protocols import LLMProviderProtocol

logger = logging.getLogger(__name__)


class ReActStepPlanner(StepPlannerProtocol):
    """Reasoning step planner generating structured proposed actions via an LLM provider.

    Adheres to StepPlannerProtocol:
    - Formats system/user prompt with tool schemas and compact observation scratchpad.
    - Queries LLM provider with token tracking and generation bounds.
    - Parses JSON output using StructuredPlannerParser.
    - Employs a single graceful repair attempt if the output fails structural validation.
    - Safely falls back to an ABSTAIN action if parsing remains unrecoverable.
    """

    def __init__(
        self,
        llm_provider: LLMProviderProtocol,
        parser: Optional[StructuredPlannerParser] = None,
        budget_tracker: Optional[ExecutionBudgetTrackerProtocol] = None,
        max_repair_attempts: int = 1,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.llm_provider = llm_provider
        self.parser = parser or StructuredPlannerParser()
        self.budget_tracker = budget_tracker
        self.max_repair_attempts = max_repair_attempts
        self.temperature = temperature
        self.max_tokens = max_tokens

    def plan_next_step(
        self,
        query: str,
        history: List[AgentStep],
        available_tools: List[ToolProtocol],
        budget_tracker: Optional[ExecutionBudgetTrackerProtocol] = None,
    ) -> ProposedAction:
        """Generate next reasoning thought and proposed action.

        Args:
            query: User's academic question.
            history: Complete sequence of past AgentSteps.
            available_tools: Registered tools available to the agent.
            budget_tracker: Optional per-run execution budget tracker.

        Returns:
            Validated ProposedAction object.
        """
        effective_budget = budget_tracker or self.budget_tracker
        tool_schemas: List[Dict[str, Any]] = [t.get_schema() for t in available_tools]
        scratchpad = format_scratchpad(history)

        budget_status: Optional[str] = None
        if effective_budget:
            consumption = effective_budget.get_consumption()
            budget_status = (
                f"Iterations: {consumption.iterations} | "
                f"Tokens: {consumption.token_usage.total_tokens} | "
                f"Exhausted: {consumption.limit_exceeded}"
            )

        messages = build_planner_prompt(
            query=query,
            available_tools_schema=tool_schemas,
            scratchpad=scratchpad,
            budget_status=budget_status,
        )

        # Compute pre-flight token budget allowance
        allowed_tokens = self.max_tokens
        if effective_budget:
            consumption = effective_budget.get_consumption()
            tracker_config = getattr(effective_budget, "config", None)
            if tracker_config is not None and hasattr(tracker_config, "max_total_tokens"):
                remaining_tokens = max(0, tracker_config.max_total_tokens - consumption.token_usage.total_tokens)
                allowed_tokens = max(16, min(self.max_tokens, remaining_tokens))

        config = LLMConfig(
            temperature=self.temperature,
            max_new_tokens=allowed_tokens,
        )

        # First generation attempt
        response = self.llm_provider.generate(messages, config=config)

        if effective_budget and response.usage:
            try:
                effective_budget.record_tokens(
                    prompt=response.usage.prompt_tokens,
                    completion=response.usage.completion_tokens,
                    is_exact=True,
                )
            except Exception as token_exc:
                logger.warning("Token budget exceeded during planner generation: %s", token_exc)
                return ProposedAction(
                    action_rationale=f"Token budget exhausted during reasoning: {token_exc}",
                    decision_summary="Abstaining due to token budget exhaustion.",
                    action_type=ActionType.ABSTAIN,
                    refusal_reason=str(token_exc),
                )

        # Parse output
        try:
            return self.parser.parse(response.content)
        except StructuralValidationError as err:
            logger.warning("Planner output failed structural validation: %s. Initiating repair.", err)

            if self.max_repair_attempts > 0:
                repair_allowed_tokens = self.max_tokens
                if effective_budget:
                    repair_consumption = effective_budget.get_consumption()
                    tracker_config = getattr(effective_budget, "config", None)
                    if tracker_config is not None and hasattr(tracker_config, "max_total_tokens"):
                        repair_remaining = max(
                            0, tracker_config.max_total_tokens - repair_consumption.token_usage.total_tokens
                        )
                        repair_allowed_tokens = max(16, min(self.max_tokens, repair_remaining))

                repair_config = LLMConfig(
                    temperature=self.temperature,
                    max_new_tokens=repair_allowed_tokens,
                )
                repair_messages = build_repair_prompt(
                    malformed_output=response.content,
                    error_message=str(err),
                )
                repair_response = self.llm_provider.generate(repair_messages, config=repair_config)

                if effective_budget and repair_response.usage:
                    try:
                        effective_budget.record_tokens(
                            prompt=repair_response.usage.prompt_tokens,
                            completion=repair_response.usage.completion_tokens,
                            is_exact=True,
                        )
                    except Exception as repair_token_exc:
                        logger.warning("Token budget exceeded during planner repair: %s", repair_token_exc)
                        return ProposedAction(
                            action_rationale=f"Token budget exhausted during reasoning repair: {repair_token_exc}",
                            decision_summary="Abstaining due to token budget exhaustion during repair.",
                            action_type=ActionType.ABSTAIN,
                            refusal_reason=str(repair_token_exc),
                        )

                try:
                    return self.parser.parse(repair_response.content)
                except StructuralValidationError as repair_err:
                    logger.error("Planner repair attempt failed: %s. Falling back to ABSTAIN.", repair_err)
                    return ProposedAction(
                        action_rationale=f"Structural validation failure in planner output: {repair_err}",
                        decision_summary="Forced abstention due to unparseable planner response.",
                        action_type=ActionType.ABSTAIN,
                        refusal_reason=f"Internal reasoning parsing failure: {repair_err}",
                    )

            # If no repair attempts permitted, return fallback ABSTAIN
            return ProposedAction(
                action_rationale=f"Structural validation failure in planner output: {err}",
                decision_summary="Forced abstention due to unparseable planner response.",
                action_type=ActionType.ABSTAIN,
                refusal_reason=f"Internal reasoning parsing failure: {err}",
            )
