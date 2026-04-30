from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from .utils import parse_float

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_precipitation_probability(
    *,
    latitude: float | None,
    longitude: float | None,
    game_datetime_utc: datetime | None,
    game_window_hours: float = 4.0,
) -> tuple[float | None, str | None]:
    if latitude is None or longitude is None or game_datetime_utc is None:
        return None, "Missing venue coordinates or game time for weather lookup"
    game_time = game_datetime_utc.astimezone(timezone.utc)
    game_end = game_time + timedelta(hours=game_window_hours)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "precipitation_probability",
        "timezone": "UTC",
        "start_date": game_time.date().isoformat(),
        "end_date": game_end.date().isoformat(),
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return None, f"Weather lookup failed: {type(exc).__name__}: {exc}"

    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    probabilities = hourly.get("precipitation_probability") or []
    if not times or not probabilities:
        return None, "Weather lookup returned no precipitation probability"

    game_window_values = []
    nearest_value = None
    nearest_delta = None
    for time_text, probability in zip(times, probabilities):
        try:
            hour = datetime.fromisoformat(str(time_text)).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        value = parse_float(probability)
        if value is None:
            continue
        # Open-Meteo's hourly precipitation probability is for the preceding hour.
        if game_time < hour <= game_end:
            game_window_values.append(value)
        delta = abs((hour - game_time).total_seconds())
        if nearest_delta is None or delta < nearest_delta:
            nearest_delta = delta
            nearest_value = value
    if game_window_values:
        return max(game_window_values), None
    return nearest_value, None if nearest_value is not None else "Weather lookup had no parseable probability"
