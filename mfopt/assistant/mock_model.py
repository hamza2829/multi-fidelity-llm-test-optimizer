"""Medium fidelity: a mock "small local model".

Two parts:
1. A keyword-scoring classifier with fuzzy token correction (difflib). It is
   more robust than the rule parser: it handles paraphrases, some indirect
   requests and many typos.
2. Controlled noise. With a probability that depends on `error_rate`, the
   difficulty of the input (estimated from the text, not from the test tags)
   and a seed, it corrupts its own answer: a confused action, a dropped or
   wrong slot, an extra slot, or broken JSON. This imitates the random
   mistakes a small model makes. Different seeds give different runs,
   which we use as "repeated runs" of the expensive level.

No network access and no model weights are needed.
"""

import difflib
import hashlib
import json
import random
import re

from .base import Assistant
from .lexicon import SLOT_LEXICON, find_number, find_value, find_zone

ACTION_KEYWORDS = {
    "set_temperature": {"temperature": 2, "temperatur": 2, "degrees": 2, "grad": 2, "heat": 2,
                        "warm": 1.5, "cold": 1, "kalt": 1, "freezing": 1, "heiz": 2},
    "set_fan_speed": {"fan": 2, "blower": 2, "gebläse": 2, "lüftung": 2, "air": 1.5,
                      "stuffy": 1},
    "set_volume": {"volume": 2, "lautstärke": 2, "loud": 1.5, "sound": 1.5},
    "navigate": {"navigate": 2, "navigiere": 2, "take": 1, "route": 1.5, "find": 1,
                 "bring": 1, "charge": 1, "laden": 1, "tank": 1.5, "eat": 1.5, "where": 0.5},
    "get_vehicle_status": {"check": 1, "show": 1, "what": 0.5, "zeig": 1, "wie": 0.5,
                           "far": 1.5, "weit": 1.5, "tires": 1.5},
    "play_media": {"play": 2, "spiele": 2, "music": 1.5, "musik": 1.5, "hören": 1.5,
                   "put": 0.5},
    "call_contact": {"call": 2, "phone": 2, "ring": 1.5, "anrufen": 2, "rufe": 2},
    "window_control": {"window": 2, "windows": 2, "fenster": 2, "open": 1, "close": 1,
                       "öffne": 1, "roll": 1, "shut": 1},
    "unsupported": {"joke": 3, "witz": 3, "pizza": 3, "alarm": 3, "flight": 3, "flug": 3,
                    "dishwasher": 3, "meaning": 3, "bestelle": 2},
}
# Slot words that also count as evidence for an action.
SLOT_ACTION = {"destination": "navigate", "item": "get_vehicle_status", "genre": "play_media",
               "contact": "call_contact", "window": "window_control"}
# Weak semantic hints the "model" has learned for indirect requests.
DEST_HINTS = {"charge": "charging_station", "laden": "charging_station",
              "tank": "gas_station", "eat": "restaurant"}
ITEM_HINTS = {"tires": "tire_pressure", "far": "range", "weit": "range"}
# Which wrong action a noisy answer drifts to.
CONFUSION = {
    "set_temperature": "set_fan_speed", "set_fan_speed": "set_temperature",
    "set_volume": "play_media", "play_media": "set_volume",
    "navigate": "get_vehicle_status", "get_vehicle_status": "navigate",
    "call_contact": "unsupported", "window_control": "set_fan_speed",
    "unsupported": "navigate",
}
OPEN_WORDS = {"open", "öffne", "roll", "auf"}
WORD_RE = re.compile(r"[a-zäöüß0-9']+")


def _vocabulary():
    words = set()
    for keywords in ACTION_KEYWORDS.values():
        words.update(keywords)
    for values in SLOT_LEXICON.values():
        for surfaces in values.values():
            for surface in surfaces:
                words.update(surface.split())
    return sorted(words)


VOCAB = _vocabulary()


class MockModelAssistant(Assistant):
    fidelity = "medium"

    def __init__(self, seed=0, error_rate=0.08):
        self.seed = seed
        self.error_rate = error_rate

    # ---- clean prediction -------------------------------------------------
    @staticmethod
    def correct_tokens(text):
        tokens = []
        for tok in WORD_RE.findall(text.lower()):
            if len(tok) >= 4 and tok not in VOCAB and not tok.isdigit():
                match = difflib.get_close_matches(tok, VOCAB, n=1, cutoff=0.8)
                if match:
                    tok = match[0]
            tokens.append(tok)
        return tokens

    def clean_predict(self, utterance):
        tokens = self.correct_tokens(utterance)
        text = " ".join(tokens)
        scores = {action: 0.0 for action in ACTION_KEYWORDS}
        for action, keywords in ACTION_KEYWORDS.items():
            scores[action] += sum(keywords.get(tok, 0) for tok in tokens)
        for slot, action in SLOT_ACTION.items():
            if find_value(text, slot):
                scores[action] += 1.5
        action = max(scores, key=lambda a: (scores[a], a == "unsupported"))
        if scores[action] < 1.0:
            action = "unsupported"
        return action, self._slots(action, text, tokens)

    @staticmethod
    def _slots(action, text, tokens):
        slots = {}
        if action == "set_temperature":
            slots["temperature"] = find_number(text)
            slots["zone"] = find_zone(text)
        elif action in ("set_fan_speed", "set_volume"):
            slots["level"] = find_number(text)
        elif action == "navigate":
            slots["destination"] = find_value(text, "destination") or next(
                (DEST_HINTS[t] for t in tokens if t in DEST_HINTS), None)
        elif action == "get_vehicle_status":
            slots["item"] = find_value(text, "item") or next(
                (ITEM_HINTS[t] for t in tokens if t in ITEM_HINTS), None)
        elif action == "play_media":
            slots["genre"] = find_value(text, "genre")
        elif action == "call_contact":
            slots["contact"] = find_value(text, "contact")
        elif action == "window_control":
            slots["window"] = find_value(text, "window")
            slots["position"] = "open" if OPEN_WORDS & set(tokens) else "close"
        return {k: v for k, v in slots.items() if v is not None}

    # ---- noise ------------------------------------------------------------
    def _rng(self, utterance):
        digest = hashlib.sha256(f"{self.seed}|{utterance}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    @staticmethod
    def _input_difficulty(utterance, tokens):
        """Longer inputs and inputs with unknown words are harder for the model."""
        unknown = sum(1 for t in tokens if t not in VOCAB and not t.isdigit())
        return 0.5 + 0.1 * len(tokens) + 0.2 * unknown

    def predict(self, utterance):
        action, slots = self.clean_predict(utterance)
        rng = self._rng(utterance)
        tokens = self.correct_tokens(utterance)
        p_error = min(0.9, self.error_rate * self._input_difficulty(utterance, tokens))
        if rng.random() >= p_error:
            return self.to_json(action, slots)

        kind = rng.choice(["action", "drop", "value", "extra", "json"])
        if kind == "action":
            action = CONFUSION[action]
        elif kind == "drop" and slots:
            slots.pop(rng.choice(sorted(slots)))
        elif kind == "value" and slots:
            key = rng.choice(sorted(slots))
            value = slots[key]
            if value.isdigit():
                slots[key] = str(int(value) + rng.choice([-2, -1, 1, 2]))
            else:
                for values in SLOT_LEXICON.values():
                    if value in values:
                        slots[key] = rng.choice([v for v in values if v != value])
                        break
        elif kind == "extra":
            slots["zone"] = "driver"
        elif kind == "json":
            return json.dumps({"action": action, "slots": slots})[:-3]
        return self.to_json(action, slots)
