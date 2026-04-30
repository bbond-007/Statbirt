from __future__ import annotations

import re

from bs4 import BeautifulSoup
import requests

from .utils import normalize_name, parse_int

LINEUPS_URL = "https://www.mlb.com/starting-lineups"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _extract_player_id_from_href(href: str) -> int | None:
    match = re.search(r"-(\d+)$", str(href or "").rstrip("/"))
    return int(match.group(1)) if match else None


def _parse_lineup_list(team_list):
    players_by_id = {}
    players_by_name = {}
    if team_list is None:
        return False, players_by_id, players_by_name
    player_items = team_list.select("li.starting-lineups__player")
    for slot, item in enumerate(player_items, start=1):
        link = item.select_one("a.starting-lineups__player--link")
        if link is None:
            continue
        player_name = normalize_name(link.get_text(" ", strip=True))
        player_id = _extract_player_id_from_href(link.get("href", ""))
        if player_id is not None:
            players_by_id[player_id] = slot
        if player_name:
            players_by_name[player_name] = slot
    return bool(player_items), players_by_id, players_by_name


def parse_confirmed_lineups_html(html: str) -> dict[int, dict]:
    output = {}
    soup = BeautifulSoup(html, "html.parser")
    for matchup in soup.select("div.starting-lineups__matchup"):
        team_links = matchup.select("div.starting-lineups__team-names a.starting-lineups__team-name--link")
        if len(team_links) < 2:
            continue
        away_team_id = parse_int(team_links[0].get("data-id"))
        home_team_id = parse_int(team_links[1].get("data-id"))
        desktop_lineups = matchup.select_one(
            "div.starting-lineups__teams.starting-lineups__teams--sm.starting-lineups__teams--xl"
        )
        lineup_scope = desktop_lineups or matchup
        for side, team_id in (("away", away_team_id), ("home", home_team_id)):
            if team_id is None:
                continue
            team_list = lineup_scope.select_one(f"ol.starting-lineups__team--{side}")
            confirmed, players_by_id, players_by_name = _parse_lineup_list(team_list)
            output[team_id] = {
                "confirmed": confirmed,
                "players_by_id": players_by_id,
                "players_by_name": players_by_name,
            }
    return output


def fetch_confirmed_lineups() -> dict[int, dict]:
    try:
        response = requests.get(LINEUPS_URL, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except Exception:
        return {}
    return parse_confirmed_lineups_html(response.text)

