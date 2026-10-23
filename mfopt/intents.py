"""Intent schema for the simulated in-car voice assistant.

An intent is a JSON object: {"action": str, "slots": {name: value}}.
Slot values are compared as lowercase strings, so 21 and "21" are equal.
"""

ACTIONS = {
    "set_temperature": {"category": "climate", "slots": ["temperature", "zone"]},
    "set_fan_speed": {"category": "climate", "slots": ["level"]},
    "navigate": {"category": "navigation", "slots": ["destination"]},
    "get_vehicle_status": {"category": "vehicle_info", "slots": ["item"]},
    "play_media": {"category": "media", "slots": ["genre"]},
    "set_volume": {"category": "media", "slots": ["level"]},
    "call_contact": {"category": "phone", "slots": ["contact"]},
    "window_control": {"category": "windows", "slots": ["window", "position"]},
    "unsupported": {"category": "out_of_scope", "slots": []},
}

CATEGORIES = sorted({spec["category"] for spec in ACTIONS.values()})

# Canonical slot values. The parsers know these catalogs, like a real
# assistant knows its POI types and contact list.
DESTINATIONS = ["charging_station", "gas_station", "home", "work", "parking", "restaurant"]
STATUS_ITEMS = ["tire_pressure", "battery_level", "range", "oil_level"]
GENRES = ["jazz", "rock", "pop", "classical", "news"]
CONTACTS = ["anna", "max", "lena", "david", "mom"]
WINDOWS = ["driver", "passenger", "all"]
ZONES = ["driver", "passenger"]
POSITIONS = ["open", "close"]

# Order in which a failing prediction is assigned its primary error type.
ERROR_TYPES = ["invalid_output", "wrong_action", "missing_slot", "wrong_slot_value", "extra_slot"]


def category_of(action):
    spec = ACTIONS.get(action)
    return spec["category"] if spec else "unknown"


def normalize_slots(slots):
    """Return slots as {name: lowercase string}, dropping empty values."""
    if not isinstance(slots, dict):
        return {}
    out = {}
    for key, value in slots.items():
        if value is None or value == "":
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        out[str(key).lower()] = str(value).strip().lower()
    return out
