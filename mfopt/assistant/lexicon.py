"""Slot value catalogs and helpers shared by the low and medium levels."""

import re

# canonical value -> surface strings (lowercase, English and German)
SLOT_LEXICON = {
    "destination": {
        "charging_station": ["charger", "charging station", "ladestation", "ladesäule"],
        "gas_station": ["gas station", "petrol station", "tankstelle"],
        "home": ["home", "nach hause"],
        "work": ["work", "arbeit"],
        "parking": ["parking", "parkplatz", "parkhaus"],
        "restaurant": ["restaurant"],
    },
    "item": {
        "tire_pressure": ["tire pressure", "tyre pressure", "reifendruck"],
        "battery_level": ["battery", "akkustand", "batteriestand"],
        "range": ["range", "reichweite"],
        "oil_level": ["oil level", "ölstand"],
    },
    "genre": {
        "jazz": ["jazz"],
        "rock": ["rock"],
        "pop": ["pop"],
        "classical": ["classical", "klassik"],
        "news": ["news", "nachrichten"],
    },
    "contact": {
        "anna": ["anna"],
        "max": ["max"],
        "lena": ["lena"],
        "david": ["david"],
        "mom": ["mom", "mother", "mama", "mutter"],
    },
    "window": {
        "driver": ["driver window", "fahrerfenster"],
        "passenger": ["passenger window", "beifahrerfenster"],
        "all": ["all windows", "alle fenster"],
    },
}

NUMBER_RE = re.compile(r"\b(\d{1,2})\b")


def find_value(text, slot):
    """Return the first canonical value whose surface string occurs in text."""
    best = None
    for value, surfaces in SLOT_LEXICON[slot].items():
        for surface in surfaces:
            pos = text.find(surface)
            if pos >= 0 and (best is None or pos < best[0]):
                best = (pos, value)
    return best[1] if best else None


def find_number(text):
    match = NUMBER_RE.search(text)
    return match.group(1) if match else None


def find_zone(text):
    # check "beifahrer" before "fahrer", it contains it
    if "passenger" in text or "beifahrer" in text:
        return "passenger"
    if "driver" in text or "fahrer" in text:
        return "driver"
    return None
