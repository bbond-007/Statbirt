from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from .utils import parse_float

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherForecast:
    precipitation_probability: float | None = None
    temperature_f: float | None = None


def fetch_weather_forecast(
    *,
    latitude: float | None,
    longitude: float | None,
    game_datetime_utc: datetime | None,
    game_window_hours: float = 4.0,
) -> tuple[WeatherForecast, str | None]:
    if latitude is None or longitude is None or game_datetime_utc is None:
        return WeatherForecast(), "Missing venue coordinates or game time for weather lookup"
    game_time = game_datetime_utc.astimezone(timezone.utc)
    game_end = game_time + timedelta(hours=game_window_hours)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "precipitation_probability,temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "start_date": game_time.date().isoformat(),
        "end_date": game_end.date().isoformat(),
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return WeatherForecast(), f"Weather lookup failed: {type(exc).__name__}: {exc}"

    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    probabilities = hourly.get("precipitation_probability") or []
    temperatures = hourly.get("temperature_2m") or []
    if not times or (not probabilities and not temperatures):
        return WeatherForecast(), "Weather lookup returned no precipitation probability or temperature"

    game_window_values = []
    nearest_value = None
    nearest_delta = None
    nearest_temperature = None
    nearest_temperature_delta = None
    for index, time_text in enumerate(times):
        try:
            hour = datetime.fromisoformat(str(time_text)).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        delta = abs((hour - game_time).total_seconds())
        probability = probabilities[index] if index < len(probabilities) else None
        value = parse_float(probability)
        if value is not None:
            # Open-Meteo's hourly precipitation probability is for the preceding hour.
            if game_time < hour <= game_end:
                game_window_values.append(value)
            if nearest_delta is None or delta < nearest_delta:
                nearest_delta = delta
                nearest_value = value
        temperature = temperatures[index] if index < len(temperatures) else None
        temperature_value = parse_float(temperature)
        if temperature_value is not None and (
            nearest_temperature_delta is None or delta < nearest_temperature_delta
        ):
            nearest_temperature_delta = delta
            nearest_temperature = temperature_value

    probability = max(game_window_values) if game_window_values else nearest_value
    forecast = WeatherForecast(
        precipitation_probability=probability,
        temperature_f=nearest_temperature,
    )
    if probability is None:
        return forecast, "Weather lookup had no parseable probability"
    return forecast, None


def fetch_precipitation_probability(
    *,
    latitude: float | None,
    longitude: float | None,
    game_datetime_utc: datetime | None,
    game_window_hours: float = 4.0,
) -> tuple[float | None, str | None]:
    forecast, warning = fetch_weather_forecast(
        latitude=latitude,
        longitude=longitude,
        game_datetime_utc=game_datetime_utc,
        game_window_hours=game_window_hours,
    )
    return forecast.precipitation_probability, warning
