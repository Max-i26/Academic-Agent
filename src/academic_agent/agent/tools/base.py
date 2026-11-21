"""Base class and common infrastructure for academic agent tools."""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError

from academic_agent.agent.models import ToolErrorCategory, ToolResult, ToolSideEffectPolicy
from academic_agent.agent.protocols import ToolProtocol


class BaseAgentTool(ToolProtocol, ABC):
    """Abstract base class for all agent tools providing declarative validation and schemas.

    Subclasses declare:
    - name: Unique tool name.
    - description: Academic and functional description.
    - side_effect_policy: ToolSideEffectPolicy instance.
    - args_model: Pydantic model defining argument structure and constraints.
    """

    name: str
    description: str
    side_effect_policy: ToolSideEffectPolicy
    args_model: Type[BaseModel]

    def get_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema from args_model."""
        if not hasattr(self, "args_model") or self.args_model is None:
            return {"type": "object", "properties": {}}
        return self.args_model.model_json_schema()

    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate argument dictionary against args_model.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        if not hasattr(self, "args_model") or self.args_model is None:
            return True, None
        try:
            self.args_model.model_validate(arguments)
            return True, None
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected validation failure: {str(e)}"

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute tool with argument validation and error containment."""
        start_time = time.perf_counter()

        # Schema validation
        is_valid, validation_err = self.validate_arguments(arguments)
        if not is_valid:
            return ToolResult(
                call_id="",
                tool_name=self.name,
                success=False,
                error_message=f"Validation failed: {validation_err}",
                error_category=ToolErrorCategory.VALIDATION_ERROR,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        try:
            validated = self.args_model.model_validate(arguments)
            output = self._run(validated)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return ToolResult(
                call_id="",
                tool_name=self.name,
                success=True,
                output=output,
                latency_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return ToolResult(
                call_id="",
                tool_name=self.name,
                success=False,
                error_message=str(exc),
                error_category=ToolErrorCategory.EXECUTION_ERROR,
                latency_ms=elapsed_ms,
            )

    @abstractmethod
    def _run(self, args: Any) -> Any:
        """Execute domain logic using validated arguments model."""
        pass
