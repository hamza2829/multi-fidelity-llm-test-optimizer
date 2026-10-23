"""Common interface for all fidelity levels."""

import json
from abc import ABC, abstractmethod


class HighFidelityUnavailable(RuntimeError):
    """Raised when the LLM level has no API access and no cached answer."""


class Assistant(ABC):
    fidelity = "undefined"

    @abstractmethod
    def predict(self, utterance):
        """Return the raw model output: a JSON string like {"action": ..., "slots": {...}}."""

    @staticmethod
    def to_json(action, slots):
        return json.dumps({"action": action, "slots": slots}, ensure_ascii=False)
