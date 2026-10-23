"""High fidelity: an optional real LLM call, with a response cache.

Disabled by default. To enable it, set both:
    MFOPT_ENABLE_LLM=1
    ANTHROPIC_API_KEY=<your key>
Optional: MFOPT_LLM_MODEL (default below).

Every answer is stored in data/llm_cache.json, keyed by model, prompt and
utterance. Later runs read from the cache, so results are repeatable and a
cached run needs no key. Without a key and without a cache entry, predict()
raises HighFidelityUnavailable and the pipeline continues with low and medium.
"""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from ..intents import ACTIONS
from .base import Assistant, HighFidelityUnavailable
from .lexicon import SLOT_LEXICON

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "llm_cache.json"
DEFAULT_MODEL = "claude-haiku-5-5"
API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = (
    "You are the intent parser of an in-car voice assistant. The user speaks English or "
    "German, and the text may contain speech recognition typos. Reply with one JSON object "
    "and nothing else: {\"action\": <action>, \"slots\": {...}}. Only include slots that the "
    "user actually stated or clearly implied. Use \"unsupported\" with empty slots for "
    "anything outside the actions below.\n"
    "Actions and slots: " + json.dumps({a: s["slots"] for a, s in ACTIONS.items()}) + "\n"
    "Allowed slot values: " + json.dumps({k: sorted(v) for k, v in SLOT_LEXICON.items()})
    + ", zone: [driver, passenger], position: [open, close], temperature and level: integers."
)


class LLMAssistant(Assistant):
    fidelity = "high"

    def __init__(self, enabled=None, model=None, cache_path=DEFAULT_CACHE):
        if enabled is None:
            enabled = os.environ.get("MFOPT_ENABLE_LLM") == "1"
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.enabled = bool(enabled and self.api_key)
        self.model = model or os.environ.get("MFOPT_LLM_MODEL", DEFAULT_MODEL)
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _key(self, utterance):
        raw = f"{self.model}|{SYSTEM_PROMPT}|{utterance}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def is_available_for(self, utterances):
        return self.enabled or all(self._key(u) in self.cache for u in utterances)

    def predict(self, utterance):
        key = self._key(utterance)
        if key in self.cache:
            return self.cache[key]["output"]
        if not self.enabled:
            raise HighFidelityUnavailable(
                "LLM level disabled (set MFOPT_ENABLE_LLM=1 and ANTHROPIC_API_KEY) "
                "and no cached answer for: " + utterance)
        output, usage = self._call_api(utterance)
        self.cache[key] = {"utterance": utterance, "model": self.model,
                           "output": output, "usage": usage}
        self._save_cache()
        return output

    def _call_api(self, utterance):
        body = json.dumps({
            "model": self.model,
            "max_tokens": 200,
            "temperature": 0,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": utterance}],
        }).encode("utf-8")
        request = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        })
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return text.strip(), data.get("usage", {})

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
