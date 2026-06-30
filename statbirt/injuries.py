from __future__ import annotations

from datetime import date
from functools import lru_cache

from .mlb_api import MLBClient
from .utils import parse_int

INJURED_STATUS_CODES = {"D7", "D10", "D15", "D60"}


def is_injured_status(status: dict | None) -> bool:
    if not isinstance(status, dict):
        return False
    code = str(status.get("code") or "").strip().upper()
    description = str(status.get("description") or "").strip().lower()
    return code in INJURED_STATUS_CODES or "injured" in description


def injured_player_ids_from_roster(roster: list[dict]) -> set[int]:
    injured: set[int] = set()
    for entry in roster:
        if not is_injured_status(entry.get("status") or {}):
            continue
        player_id = parse_int((entry.get("person") or {}).get("id"))
        if player_id is not None:
            injured.add(player_id)
    return injured


@lru_cache(maxsize=16)
def fetch_current_injured_player_ids(team_ids: tuple[int, ...], season: int) -> frozenset[int]:
    client = MLBClient()
    injured: set[int] = set()
    for team_id in sorted(set(team_ids)):
        try:
            roster = client.roster(team_id, roster_type="fullSeason", season=season)
        except Exception:
            continue
        injured.update(injured_player_ids_from_roster(roster))
    return frozenset(injured)


def current_injured_player_ids_for_rows(rows: list[dict[str, str]], *, season: int | None = None) -> set[int]:
    team_ids = tuple(
        sorted(
            {
                team_id
                for team_id in (parse_int(row.get("team_id")) for row in rows)
                if team_id is not None
            }
        )
    )
    if not team_ids:
        return set()
    return set(fetch_current_injured_player_ids(team_ids, season or date.today().year))


def is_injured_player(row: dict[str, str], injured_player_ids: set[int] | frozenset[int]) -> bool:
    player_id = parse_int(row.get("player_id"))
    return player_id is not None and player_id in injured_player_ids


def filter_injured_rows(
    rows: list[dict[str, str]],
    injured_player_ids: set[int] | frozenset[int],
) -> tuple[list[dict[str, str]], int]:
    if not injured_player_ids:
        return rows, 0
    filtered = [row for row in rows if not is_injured_player(row, injured_player_ids)]
    return filtered, len(rows) - len(filtered)
