"""Low fidelity: a rule-based parser. Instant and free, but brittle.

It only knows canonical command phrasings. Paraphrases, indirect requests and
typos usually end up as "unsupported" or with missing slots.
"""

import re

from .base import Assistant
from .lexicon import find_number, find_value, find_zone

RULES = [
    ("set_temperature", re.compile(r"\b(set|stelle|stell)\b.*\btemperatur(e)?\b")),
    ("set_fan_speed", re.compile(r"\b(fan speed|gebläse)\b")),
    ("set_volume", re.compile(r"\b(volume|lautstärke)\b")),
    ("navigate", re.compile(r"^(navigate to|take me to|navigiere|bring mich)\b")),
    ("get_vehicle_status", re.compile(r"^(what is my|show|wie ist der|zeig mir)\b")),
    ("play_media", re.compile(r"^(play|spiele)\b")),
    ("call_contact", re.compile(r"^(call|rufe)\b")),
    ("window_control", re.compile(r"^(open|close|öffne)\b")),
]


class RuleBasedAssistant(Assistant):
    fidelity = "low"

    def predict(self, utterance):
        text = utterance.lower().strip()
        for action, pattern in RULES:
            if pattern.search(text):
                return self.to_json(action, self._slots(action, text))
        return self.to_json("unsupported", {})

    @staticmethod
    def _slots(action, text):
        slots = {}
        if action == "set_temperature":
            slots["temperature"] = find_number(text)
            slots["zone"] = find_zone(text)
        elif action in ("set_fan_speed", "set_volume"):
            slots["level"] = find_number(text)
        elif action == "navigate":
            slots["destination"] = find_value(text, "destination")
        elif action == "get_vehicle_status":
            slots["item"] = find_value(text, "item")
        elif action == "play_media":
            slots["genre"] = find_value(text, "genre")
        elif action == "call_contact":
            slots["contact"] = find_value(text, "contact")
        elif action == "window_control":
            slots["window"] = find_value(text, "window")
            slots["position"] = "close" if text.startswith("close") else "open"
        return {k: v for k, v in slots.items() if v is not None}
