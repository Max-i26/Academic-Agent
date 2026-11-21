"""Multi-dimensional budget enforcement engine for bounded agent autonomy."""

import json
import threading
import time
from typing import Optional, Set, Tuple

from academic_agent.agent.models import (
    BudgetConfig,
    BudgetConsumption,
    TokenAccounting,
    ToolCall,
)
from academic_agent.agent.protocols import ExecutionBudgetTrackerProtocol


class BudgetExceededError(Exception):
    """Raised when an execution ceiling (iterations, timeout, tokens, tools) is breached."""

    def __init__(self, resource: str, limit: float, consumed: float, message: Optional[str] = None):
        msg = message or f"Execution budget exceeded for {resource}: consumed {consumed} exceeds limit {limit}"
        super().__init__(msg)
        self.resource = resource
        self.limit = limit
        self.consumed = consumed


class ExecutionBudgetTracker(ExecutionBudgetTrackerProtocol):
    """Enforces multi-dimensional resource bounds across agent execution.

    Thread Safety:
    - All state mutation and reading operations are synchronized via an internal RLock,
      preventing race conditions during concurrent tool or planning calls.

    Timeout Semantics:
    - This tracker provides cooperative checkpoint timeout enforcement at step and tool
      dispatch boundaries (via check_time() and record_*()).
    - Sub-second active interruption of individual tool calls is delegated to the Stage 6.2
      ToolDispatcher using tool.side_effect_policy.timeout_seconds and thread/process boundaries.
    - Global run deadlines (get_remaining_time()) are propagated to provider LLM generation calls.
    """

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        start_time: Optional[float] = None,
    ) -> None:
        self.config: BudgetConfig = config or BudgetConfig()
        self.start_time: float = start_time if start_time is not None else time.perf_counter()
        self.iterations: int = 0
        self.tool_invocations: int = 0
        self.token_usage: TokenAccounting = TokenAccounting(prompt_tokens=0, completion_tokens=0, total_tokens=0, is_exact=True)
        self._seen_tool_calls: Set[Tuple[str, str]] = set()
        self.duplicate_calls_detected: int = 0
        self._exhausted_resource: Optional[str] = None
        self._lock = threading.RLock()

    @property
    def deadline(self) -> float:
        """Absolute monotonic deadline timestamp for this execution."""
        return self.start_time + self.config.timeout_seconds

    def get_remaining_time(self) -> float:
        """Return remaining execution time in seconds, bounded below by 0.0."""
        with self._lock:
            remaining = self.deadline - time.perf_counter()
            return max(0.0, remaining)

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed wall-clock execution time in seconds."""
        return time.perf_counter() - self.start_time

    def check_time(self) -> None:
        """Verify elapsed time against timeout ceiling; raises BudgetExceededError if expired."""
        with self._lock:
            elapsed = self.elapsed_seconds
            if elapsed > self.config.timeout_seconds:
                self._exhausted_resource = "timeout"
                raise BudgetExceededError(
                    resource="timeout_seconds",
                    limit=self.config.timeout_seconds,
                    consumed=round(elapsed, 2),
                )

    def record_iteration(self) -> None:
        """Record completed planning cycle and enforce iteration ceiling."""
        with self._lock:
            self.check_time()
            self.iterations += 1
            if self.iterations > self.config.max_iterations:
                self._exhausted_resource = "iterations"
                raise BudgetExceededError(
                    resource="max_iterations",
                    limit=self.config.max_iterations,
                    consumed=self.iterations,
                )

    def record_tokens(self, prompt: int, completion: int, is_exact: bool = True) -> None:
        """Accumulate token consumption and enforce cumulative token ceiling."""
        with self._lock:
            self.check_time()
            self.token_usage = self.token_usage.add(prompt, completion, is_exact)
            if self.token_usage.total_tokens > self.config.max_total_tokens:
                self._exhausted_resource = "tokens"
                raise BudgetExceededError(
                    resource="max_total_tokens",
                    limit=self.config.max_total_tokens,
                    consumed=self.token_usage.total_tokens,
                )

    def record_tool_call(self, tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
        """Authorize tool invocation against tool limits and duplicate detection.

        Returns:
            Tuple of (authorized: bool, rejection_reason: Optional[str]).
        """
        with self._lock:
            self.check_time()

            # Check tool invocation count ceiling
            if self.tool_invocations >= self.config.max_tool_invocations:
                self._exhausted_resource = "tool_invocations"
                return False, f"Maximum tool invocations reached ({self.config.max_tool_invocations})"

            # Check duplicate tool call
            if self.config.detect_duplicate_calls:
                norm_args = json.dumps(tool_call.arguments, sort_keys=True)
                signature = (tool_call.tool_name.lower().strip(), norm_args)
                if signature in self._seen_tool_calls:
                    self.duplicate_calls_detected += 1
                    return False, f"Duplicate tool invocation rejected: '{tool_call.tool_name}' with identical parameters"
                self._seen_tool_calls.add(signature)

            self.tool_invocations += 1
            return True, None

    def is_exhausted(self) -> bool:
        """True if any budget ceiling has been marked exhausted or exceeded."""
        with self._lock:
            if self._exhausted_resource is not None:
                return True
            if self.elapsed_seconds > self.config.timeout_seconds:
                self._exhausted_resource = "timeout"
                return True
            if self.iterations >= self.config.max_iterations:
                self._exhausted_resource = "iterations"
                return True
            if self.token_usage.total_tokens >= self.config.max_total_tokens:
                self._exhausted_resource = "tokens"
                return True
            if self.tool_invocations >= self.config.max_tool_invocations:
                self._exhausted_resource = "tool_invocations"
                return True
            return False

    def get_exhausted_resource(self) -> Optional[str]:
        """Name of exhausted budget resource."""
        with self._lock:
            self.is_exhausted()
            return self._exhausted_resource

    def get_consumption(self) -> BudgetConsumption:
        """Construct an immutable report of consumed resources."""
        with self._lock:
            exhausted = self.is_exhausted()
            return BudgetConsumption(
                iterations=self.iterations,
                elapsed_seconds=round(self.elapsed_seconds, 3),
                tool_invocations=self.tool_invocations,
                token_usage=self.token_usage,
                duplicate_calls_detected=self.duplicate_calls_detected,
                limit_exceeded=exhausted,
                exhausted_resource=self._exhausted_resource,
            )
