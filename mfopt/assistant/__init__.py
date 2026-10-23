from .base import Assistant, HighFidelityUnavailable
from .llm import LLMAssistant
from .mock_model import MockModelAssistant
from .rule_based import RuleBasedAssistant


def build_assistants(seed=0, error_rate=None, enable_llm=None):
    """Return {fidelity: assistant} for the three levels."""
    medium = MockModelAssistant(seed=seed) if error_rate is None else MockModelAssistant(
        seed=seed, error_rate=error_rate)
    return {
        "low": RuleBasedAssistant(),
        "medium": medium,
        "high": LLMAssistant(enabled=enable_llm),
    }


__all__ = ["Assistant", "HighFidelityUnavailable", "LLMAssistant", "MockModelAssistant",
           "RuleBasedAssistant", "build_assistants"]
