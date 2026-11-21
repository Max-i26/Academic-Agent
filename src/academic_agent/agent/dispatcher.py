"""Tool dispatcher enforcing schema validation, budget pre-checks, hard timeouts, and error containment."""

import concurrent.futures
import time
from typing import Optional

from academic_agent.agent.models import ToolCall, ToolErrorCategory, ToolResult
from academic_agent.agent.protocols import (
    ExecutionBudgetTrackerProtocol,
    ToolDispatcherProtocol,
    ToolRegistryProtocol,
)


class ToolDispatcher(ToolDispatcherProtocol):
    """Orchestrates authorized, isolated, and bounded tool execution.

    Enforces the Three Non-Negotiable Rules of Agent Tool Execution:
    1. Schema Validation: A tool cannot execute unless its arguments pass schema validation.
    2. Boundary Isolation: Tools cannot access or mutate agent state, FSM, memory, or budget.
    3. Exception & Timeout Containment: Timeouts and crashes are converted to structured ToolResult observations.
    """

    def __init__(self, registry: ToolRegistryProtocol, max_workers: int = 4) -> None:
        self.registry = registry
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ToolWorker",
        )

    def dispatch(
        self,
        tool_call: ToolCall,
        budget_tracker: Optional[ExecutionBudgetTrackerProtocol] = None,
    ) -> ToolResult:
        """Execute a tool call with pre-authorization, hard timeout, and error containment.

        Args:
            tool_call: Immutable tool call specification containing name and arguments.
            budget_tracker: Optional multi-dimensional resource tracker for budget checking.

        Returns:
            ToolResult containing observation payload or categorized error information.
        """
        start_mono = time.perf_counter()

        # 1. Lookup Tool in Registry
        tool = self.registry.get(tool_call.tool_name)
        if tool is None:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error_message=f"Tool '{tool_call.tool_name}' is not registered in ToolRegistry.",
                error_category=ToolErrorCategory.NOT_FOUND,
                latency_ms=round((time.perf_counter() - start_mono) * 1000.0, 2),
            )

        # 2. Schema / Argument Validation (Non-Negotiable Rule 1)
        validate_fn = getattr(tool, "validate_arguments", None)
        if callable(validate_fn):
            is_valid, validation_err = validate_fn(tool_call.arguments)
            if not is_valid:
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    success=False,
                    error_message=f"Schema validation failed for tool '{tool_call.tool_name}': {validation_err}",
                    error_category=ToolErrorCategory.VALIDATION_ERROR,
                    latency_ms=round((time.perf_counter() - start_mono) * 1000.0, 2),
                )

        # 3. Budget Pre-Check and Duplicate Interception (Pre-enforcement)
        if budget_tracker is not None:
            # Check elapsed wall-clock timeout
            try:
                budget_tracker.check_time()
            except Exception as e:
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    success=False,
                    error_message=f"Execution budget exceeded before tool dispatch: {str(e)}",
                    error_category=ToolErrorCategory.TIMEOUT,
                    latency_ms=round((time.perf_counter() - start_mono) * 1000.0, 2),
                )

            # Record and authorize tool call (interception of duplicates & invocation limits)
            authorized, rejection_reason = budget_tracker.record_tool_call(tool_call)
            if not authorized:
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    success=False,
                    error_message=rejection_reason or "Tool call rejected by execution budget tracker.",
                    error_category=ToolErrorCategory.PERMISSION_DENIED,
                    latency_ms=round((time.perf_counter() - start_mono) * 1000.0, 2),
                )

        # 4. Compute Effective Monotonic Timeout (Non-Negotiable Rule 3)
        tool_timeout = tool.side_effect_policy.timeout_seconds
        if budget_tracker is not None and hasattr(budget_tracker, "get_remaining_time"):
            remaining_time = budget_tracker.get_remaining_time()
            effective_timeout = max(0.01, min(tool_timeout, remaining_time))
        else:
            effective_timeout = max(0.01, tool_timeout)

        # 5. Isolated Bounded Execution (Non-Negotiable Rules 2 & 3)
        try:
            future = self._executor.submit(tool.execute, tool_call.arguments)
            result = future.result(timeout=effective_timeout)

            # Ensure call_id, tool_name, and latency are accurately set
            elapsed_ms = round((time.perf_counter() - start_mono) * 1000.0, 2)
            if isinstance(result, ToolResult):
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    success=result.success,
                    output=result.output,
                    error_message=result.error_message,
                    error_category=result.error_category,
                    latency_ms=elapsed_ms if result.latency_ms == 0.0 else result.latency_ms,
                )
            else:
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    success=True,
                    output=result,
                    latency_ms=elapsed_ms,
                )

        except concurrent.futures.TimeoutError:
            elapsed_ms = round((time.perf_counter() - start_mono) * 1000.0, 2)
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error_message=f"Tool '{tool_call.tool_name}' timed out after {effective_timeout:.2f}s",
                error_category=ToolErrorCategory.TIMEOUT,
                latency_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_mono) * 1000.0, 2)
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error_message=f"Tool '{tool_call.tool_name}' raised unhandled exception: {str(exc)}",
                error_category=ToolErrorCategory.EXECUTION_ERROR,
                latency_ms=elapsed_ms,
            )

    def shutdown(self, wait: bool = False) -> None:
        """Release background threadpool resources."""
        self._executor.shutdown(wait=wait)
