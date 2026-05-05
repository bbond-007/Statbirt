from __future__ import annotations

import csv
from pathlib import Path

import requests

from .utils import normalize_name, parse_float, parse_int

FANGRAPHS_LEADERS_API = "https://www.fangraphs.com/api/leaders/major-league/data"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}


def _stuff_value_from_row(row: dict) -> float | None:
    for key in (
        "Stuff+",
        "stuff_plus",
        "stuff+",
        "Stf+",
        "stf_plus",
        "sp_stuff",
        "StuffPlus",
        "stuffPlus",
    ):
        value = parse_float(row.get(key))
        if value is not None:
            return value
    return None


def _name_from_row(row: dict) -> str:
    for key in ("Name", "PlayerName", "player_name", "playerName", "name", "Player"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def load_manual_stuff_plus(path: str | Path) -> tuple[dict[int, float], dict[str, float]]:
    path = Path(path)
    by_id: dict[int, float] = {}
    by_name: dict[str, float] = {}
    if not path.exists():
        return by_id, by_name
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stuff_plus = _stuff_value_from_row(row)
            if stuff_plus is None:
                continue
            player_id = parse_int(row.get("player_id") or row.get("mlbam_id") or row.get("MLBAMID"))
            if player_id is not None:
                by_id[player_id] = stuff_plus
            name = normalize_name(_name_from_row(row))
            if name:
                by_name[name] = stuff_plus
    return by_id, by_name


def fetch_fangraphs_stuff_plus(season: int, *, page_items: int = 2000) -> tuple[dict[str, float], str | None]:
    params = {
        "age": "",
        "pos": "all",
        "stats": "pit",
        "lg": "all",
        "qual": "0",
        "season": str(season),
        "season1": str(season),
        "startdate": f"{season}-03-01",
        "enddate": f"{season}-11-30",
        "month": "0",
        "hand": "",
        "team": "0",
        "pageitems": str(page_items),
        "pagenum": "1",
        "ind": "0",
        "rost": "0",
        "players": "",
        "type": "36",
        "postseason": "",
        "sortdir": "default",
        "sortstat": "Stuff+",
    }
    try:
        response = requests.get(FANGRAPHS_LEADERS_API, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {}, f"FanGraphs Stuff+ fetch failed: {type(exc).__name__}: {exc}"

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}, "FanGraphs Stuff+ fetch returned an unexpected payload shape"

    by_name = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = normalize_name(_name_from_row(row))
        stuff_plus = _stuff_value_from_row(row)
        if name and stuff_plus is not None:
            by_name[name] = stuff_plus
    return by_name, None


def build_stuff_plus_lookup(
    *,
    season: int,
    manual_csv: str | Path,
    use_fetch: bool = True,
) -> tuple[dict[int, float], dict[str, float], list[str]]:
    warnings: list[str] = []
    by_id, by_name = load_manual_stuff_plus(manual_csv)
    if use_fetch:
        fetched_by_name, warning = fetch_fangraphs_stuff_plus(season)
        if warning:
            warnings.append(warning)
        by_name = {**fetched_by_name, **by_name}
    return by_id, by_name, warnings


def lookup_stuff_plus(
    *,
    player_id: int | None,
    player_name: str,
    by_id: dict[int, float],
    by_name: dict[str, float],
) -> float | None:
    if player_id is not None and player_id in by_id:
        return by_id[player_id]
    return by_name.get(normalize_name(player_name))
