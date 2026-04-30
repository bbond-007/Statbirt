from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
import re
import unicodedata

TEAM_ABBR_BY_ID = {
    108: "LAA",
    109: "ARI",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KCR",
    119: "LAD",
    120: "WSN",
    121: "NYM",
    133: "ATH",
    134: "PIT",
    135: "SDP",
    136: "SEA",
    137: "SFG",
    138: "STL",
    139: "TBR",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CHW",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}

TEAM_ID_BY_ABBR = {abbr: team_id for team_id, abbr in TEAM_ABBR_BY_ID.items()}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
NON_AB_EVENTS = {
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_bunt",
    "sac_fly",
    "catcher_interf",
    "catchers_interference",
}
SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_bunt",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}


def parse_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except (TypeError, ValueError, OverflowError):
        return None


def safe_divide(numerator, denominator) -> float | None:
    numerator = parse_float(numerator)
    denominator = parse_float(denominator)
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def weighted_average(pairs, default=None):
    total_weight = 0.0
    total_value = 0.0
    for value, weight in pairs:
        parsed = parse_float(value)
        weight = parse_float(weight)
        if parsed is None or weight is None or weight <= 0:
            continue
        total_value += parsed * weight
        total_weight += weight
    if total_weight <= 0:
        return default
    return total_value / total_weight


def score_on_scale(
    value,
    low: float,
    high: float,
    *,
    higher_is_better: bool = True,
    default: float = 50.0,
) -> float:
    parsed = parse_float(value)
    if parsed is None or high == low:
        return default
    if higher_is_better:
        raw = (parsed - low) / (high - low)
    else:
        raw = (high - parsed) / (high - low)
    return clamp(raw, 0.0, 1.0) * 100.0


def normalize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.lower()).strip()


def last_first_to_first_last(name: str) -> str:
    text = (name or "").strip()
    if "," not in text:
        return text
    last, first = [part.strip() for part in text.split(",", 1)]
    return f"{first} {last}".strip()


def parse_mlb_innings(value) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.fullmatch(r"(\d+)(?:\.(\d))?", text)
    if not match:
        return parse_float(text)
    whole = int(match.group(1))
    partial = match.group(2) or "0"
    if partial == "0":
        return float(whole)
    if partial == "1":
        return whole + (1.0 / 3.0)
    if partial == "2":
        return whole + (2.0 / 3.0)
    return None


def format_float(value, digits: int = 3) -> str:
    parsed = parse_float(value)
    return "" if parsed is None else f"{parsed:.{digits}f}"


def team_abbr(team_id, fallback: str = "") -> str:
    parsed = parse_int(team_id)
    if parsed is None:
        return fallback
    return TEAM_ABBR_BY_ID.get(parsed, fallback)


def canonical_date(value) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

