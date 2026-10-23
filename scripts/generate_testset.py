"""Generate the synthetic test set (data/testset.json).

The output is committed to the repo. Re-running this script with the same
seed produces the same file. Utterances come from hand-written templates in
English and German, and about one in five is copied with typos to simulate
noisy speech recognition.

Usage: python scripts/generate_testset.py
"""

import json
import random
from pathlib import Path

SEED = 7
TARGET_CLEAN = 120
TARGET_NOISY = 30
OUT = Path(__file__).resolve().parent.parent / "data" / "testset.json"

# Surface forms per canonical value and language.
DEST = {
    "charging_station": {"en": ["the nearest charger", "a charging station"],
                         "de": ["zur nächsten Ladestation", "zur Ladesäule"]},
    "gas_station": {"en": ["the nearest gas station", "a petrol station"],
                    "de": ["zur nächsten Tankstelle", "zur Tankstelle"]},
    "home": {"en": ["home"], "de": ["nach Hause"]},
    "work": {"en": ["work"], "de": ["zur Arbeit"]},
    "parking": {"en": ["a parking spot", "the nearest parking garage"],
                "de": ["zum nächsten Parkplatz", "zu einem Parkhaus"]},
    "restaurant": {"en": ["a restaurant nearby"], "de": ["zu einem Restaurant"]},
}
STATUS = {
    "tire_pressure": {"en": ["tire pressure", "tyre pressure"], "de": ["Reifendruck"]},
    "battery_level": {"en": ["battery level", "battery charge"], "de": ["Akkustand", "Batteriestand"]},
    "range": {"en": ["range", "remaining range"], "de": ["Reichweite", "Restreichweite"]},
    "oil_level": {"en": ["oil level"], "de": ["Ölstand"]},
}
GENRE = {
    "jazz": {"en": ["jazz"], "de": ["Jazz"]},
    "rock": {"en": ["rock music", "rock"], "de": ["Rockmusik"]},
    "pop": {"en": ["pop music", "pop"], "de": ["Popmusik"]},
    "classical": {"en": ["classical music"], "de": ["klassische Musik", "Klassik"]},
    "news": {"en": ["the news"], "de": ["Nachrichten"]},
}
CONTACT = {
    "anna": {"en": ["Anna"], "de": ["Anna"]},
    "max": {"en": ["Max"], "de": ["Max"]},
    "lena": {"en": ["Lena"], "de": ["Lena"]},
    "david": {"en": ["David"], "de": ["David"]},
    "mom": {"en": ["mom", "my mother"], "de": ["Mama", "meine Mutter"]},
}
WINDOW = {
    "driver": {"en": ["the driver window"], "de": ["das Fahrerfenster"]},
    "passenger": {"en": ["the passenger window"], "de": ["das Beifahrerfenster"]},
    "all": {"en": ["all windows"], "de": ["alle Fenster"]},
}

# Each template: (language, difficulty, text, slot spec). Slot spec maps a slot
# name to a value source: "temp", "fan", "vol" draw numbers, a dict draws a
# canonical value and its surface form, and a plain string is a fixed value.
T = []


def add(action, lang, difficulty, text, slots=None):
    T.append({"action": action, "language": lang, "difficulty": difficulty,
              "text": text, "slots": slots or {}})


# climate: set_temperature
add("set_temperature", "en", "easy", "set temperature to {temp}", {"temperature": "temp"})
add("set_temperature", "en", "easy", "set the temperature to {temp} degrees", {"temperature": "temp"})
add("set_temperature", "en", "medium", "make it {temp} degrees in here", {"temperature": "temp"})
add("set_temperature", "en", "medium", "set the driver side to {temp} degrees",
    {"temperature": "temp", "zone": "driver"})
add("set_temperature", "en", "hard", "I'm freezing, could you bump the heat up to {temp} please",
    {"temperature": "temp"})
add("set_temperature", "en", "hard", "passenger says it's too cold, warm their side to {temp}",
    {"temperature": "temp", "zone": "passenger"})
add("set_temperature", "de", "easy", "stelle die Temperatur auf {temp} Grad", {"temperature": "temp"})
add("set_temperature", "de", "medium", "mach es bitte {temp} Grad warm", {"temperature": "temp"})
add("set_temperature", "de", "hard", "mir ist kalt, heiz auf der Beifahrerseite auf {temp} hoch",
    {"temperature": "temp", "zone": "passenger"})
# climate: set_fan_speed
add("set_fan_speed", "en", "easy", "set fan speed to {fan}", {"level": "fan"})
add("set_fan_speed", "en", "medium", "turn the blower to level {fan}", {"level": "fan"})
add("set_fan_speed", "en", "hard", "it's stuffy, put the air on {fan}", {"level": "fan"})
add("set_fan_speed", "de", "easy", "stelle das Gebläse auf Stufe {fan}", {"level": "fan"})
add("set_fan_speed", "de", "medium", "Lüftung auf {fan} bitte", {"level": "fan"})
# navigation
add("navigate", "en", "easy", "navigate to {dest}", {"destination": DEST})
add("navigate", "en", "easy", "take me to {dest}", {"destination": DEST})
add("navigate", "en", "medium", "find {dest} and start route guidance", {"destination": DEST})
add("navigate", "en", "hard", "battery is getting low, find me somewhere to charge",
    {"destination": "charging_station"})
add("navigate", "en", "hard", "I need to fill up the tank soon, where can I do that",
    {"destination": "gas_station"})
add("navigate", "en", "hard", "I'm hungry, is there somewhere to eat on the way",
    {"destination": "restaurant"})
add("navigate", "de", "easy", "navigiere {dest}", {"destination": DEST})
add("navigate", "de", "medium", "bring mich bitte {dest}", {"destination": DEST})
add("navigate", "de", "hard", "der Akku ist fast leer, wo kann ich laden",
    {"destination": "charging_station"})
# vehicle info
add("get_vehicle_status", "en", "easy", "what is my {status}", {"item": STATUS})
add("get_vehicle_status", "en", "easy", "show {status}", {"item": STATUS})
add("get_vehicle_status", "en", "medium", "can you check the {status} for me", {"item": STATUS})
add("get_vehicle_status", "en", "hard", "how much air is in my tires", {"item": "tire_pressure"})
add("get_vehicle_status", "en", "hard", "how far can I drive before I need to charge",
    {"item": "range"})
add("get_vehicle_status", "de", "easy", "wie ist der {status}", {"item": STATUS})
add("get_vehicle_status", "de", "medium", "zeig mir bitte den {status}", {"item": STATUS})
add("get_vehicle_status", "de", "hard", "wie weit komme ich noch", {"item": "range"})
# media
add("play_media", "en", "easy", "play {genre}", {"genre": GENRE})
add("play_media", "en", "medium", "put on some {genre}", {"genre": GENRE})
add("play_media", "en", "hard", "I'm bored, something with saxophones would be nice",
    {"genre": "jazz"})
add("play_media", "de", "easy", "spiele {genre}", {"genre": GENRE})
add("play_media", "de", "medium", "ich möchte {genre} hören", {"genre": GENRE})
add("set_volume", "en", "easy", "set volume to {vol}", {"level": "vol"})
add("set_volume", "en", "medium", "change the volume level to {vol}", {"level": "vol"})
add("set_volume", "en", "hard", "it's way too loud, bring the sound down to {vol}", {"level": "vol"})
add("set_volume", "de", "easy", "Lautstärke auf {vol}", {"level": "vol"})
add("set_volume", "de", "medium", "stell die Lautstärke bitte auf {vol}", {"level": "vol"})
# phone
add("call_contact", "en", "easy", "call {contact}", {"contact": CONTACT})
add("call_contact", "en", "medium", "please phone {contact}", {"contact": CONTACT})
add("call_contact", "en", "hard", "I should let {contact} know I'm running late, ring them",
    {"contact": CONTACT})
add("call_contact", "de", "easy", "rufe {contact} an", {"contact": CONTACT})
add("call_contact", "de", "medium", "kannst du bitte {contact} anrufen", {"contact": CONTACT})
# windows
add("window_control", "en", "easy", "open {window}", {"window": WINDOW, "position": "open"})
add("window_control", "en", "easy", "close {window}", {"window": WINDOW, "position": "close"})
add("window_control", "en", "medium", "roll down {window}", {"window": WINDOW, "position": "open"})
add("window_control", "en", "hard", "it's raining in, shut {window}",
    {"window": WINDOW, "position": "close"})
add("window_control", "de", "easy", "öffne {window}", {"window": WINDOW, "position": "open"})
add("window_control", "de", "medium", "mach {window} zu", {"window": WINDOW, "position": "close"})
# out of scope: the assistant must answer "unsupported"
for lang, diff, text in [
    ("en", "easy", "tell me a joke"),
    ("en", "easy", "order a pizza"),
    ("en", "medium", "what is the meaning of life"),
    ("en", "hard", "set an alarm for 7 tomorrow"),
    ("en", "hard", "turn on the dishwasher at home"),
    ("en", "hard", "book a flight and play it safe"),
    ("de", "easy", "erzähl mir einen Witz"),
    ("de", "medium", "bestelle mir eine Pizza"),
    ("de", "hard", "buche einen Flug nach Hause für morgen"),
]:
    add("unsupported", lang, diff, text)

ACTION_CATEGORY = {
    "set_temperature": "climate", "set_fan_speed": "climate", "navigate": "navigation",
    "get_vehicle_status": "vehicle_info", "play_media": "media", "set_volume": "media",
    "call_contact": "phone", "window_control": "windows", "unsupported": "out_of_scope",
}
PLACEHOLDER = {"destination": "dest", "item": "status", "genre": "genre",
               "contact": "contact", "window": "window"}


def fill(template, rng):
    text = template["text"]
    slots = {}
    for name, source in template["slots"].items():
        if source == "temp":
            value = rng.randint(16, 28)
            text = text.replace("{temp}", str(value))
        elif source == "fan":
            value = rng.randint(1, 5)
            text = text.replace("{fan}", str(value))
        elif source == "vol":
            value = rng.randint(1, 10)
            text = text.replace("{vol}", str(value))
        elif isinstance(source, dict):
            value = rng.choice(sorted(source))
            surface = rng.choice(source[value][template["language"]])
            text = text.replace("{" + PLACEHOLDER[name] + "}", surface)
        else:
            value = source
        slots[name] = str(value)
    return text, slots


def add_typos(text, rng):
    """Apply 1 or 2 character edits to words longer than 3 letters."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
    for i in rng.sample(candidates, min(len(candidates), rng.choice([1, 2]))):
        w = words[i]
        pos = rng.randrange(1, len(w) - 1)
        kind = rng.choice(["swap", "drop", "double"])
        if kind == "swap":
            w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
        elif kind == "drop":
            w = w[:pos] + w[pos + 1:]
        else:
            w = w[:pos] + w[pos] + w[pos:]
        words[i] = w
    return " ".join(words)


def costs(text, difficulty):
    tokens = round(len(text.split()) * 1.3)
    return {
        # rule parser: effectively free
        "low_s": round(0.0002 + 0.00001 * len(text), 5),
        # simulated small local model on CPU
        "medium_s": round(0.15 + 0.02 * tokens, 3),
        # remote LLM call: latency grows with input length, hard prompts get longer answers
        "high_s": round(0.9 + 0.02 * tokens + (0.3 if difficulty == "hard" else 0.0), 3),
        "high_tokens": 420 + tokens + 30,
    }


def main():
    rng = random.Random(SEED)
    clean = []
    seen = set()
    # Cycle through templates so every template appears, then keep sampling
    # until the target is reached. Duplicated texts are skipped.
    order = list(range(len(T)))
    while len(clean) < TARGET_CLEAN:
        rng.shuffle(order)
        for idx in order:
            if len(clean) >= TARGET_CLEAN:
                break
            tpl = T[idx]
            text, slots = fill(tpl, rng)
            if text in seen:
                continue
            seen.add(text)
            clean.append((tpl, text, slots, "clean"))

    noisy = []
    for tpl, text, slots, _ in rng.sample(clean, TARGET_NOISY):
        noisy.append((tpl, add_typos(text, rng), slots, "typo"))

    tests = []
    for n, (tpl, text, slots, noise) in enumerate(clean + noisy, start=1):
        tests.append({
            "id": f"t{n:03d}",
            "utterance": text,
            "expected": {"action": tpl["action"], "slots": slots},
            "tags": {
                "category": ACTION_CATEGORY[tpl["action"]],
                "difficulty": tpl["difficulty"],
                "language": tpl["language"],
                "noise": noise,
            },
            "cost": costs(text, tpl["difficulty"]),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tests, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(tests)} tests from {len(T)} templates to {OUT}")


if __name__ == "__main__":
    main()
