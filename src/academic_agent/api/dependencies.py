"""Dependency injection providers for FastAPI serving layer."""

from typing import Optional

from academic_agent.service.protocols import AgentServiceProtocol

_DEFAULT_SERVICE: Optional[AgentServiceProtocol] = None


def set_default_agent_service(service: Optional[AgentServiceProtocol]) -> None:
    """Register the global default agent service instance."""
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = service


def get_agent_service() -> AgentServiceProtocol:
    """FastAPI dependency provider returning the active AgentService instance.

    Raises:
        RuntimeError: If no service has been configured on the application.
    """
    if _DEFAULT_SERVICE is None:
        raise RuntimeError(
            "AcademicAgentService has not been initialized. "
            "Pass service to create_app(service=...) or register via app.dependency_overrides."
        )
    return _DEFAULT_SERVICE


def reset_dependencies() -> None:
    """Reset global dependency state (useful in test tear-down)."""
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = None


__all__ = [
    "get_agent_service",
    "set_default_agent_service",
    "reset_dependencies",
]
