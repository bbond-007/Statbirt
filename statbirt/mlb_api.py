from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import time

import requests

from .cache import cache_path, load_pickle, save_pickle
from .models import GameContext
from .utils import (
    canonical_date,
    parse_float,
    parse_int,
    parse_mlb_innings,
    safe_divide,
    team_abbr,
)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "Statbirt/0.1 (+https://statsapi.mlb.com)"}
PLAY_LOG_CACHE_SCHEMA = 1
PITCHER_LOG_CACHE_SCHEMA = 2
RECENT_USAGE_CACHE_SCHEMA = 1
BULLPEN_CACHE_SCHEMA = 1
BULLPEN_STATS_CACHE_SCHEMA = 2
VENUE_COORDINATE_OVERRIDES = {
    # MLB's venue endpoint includes city/country for this Mexico City venue, but not defaultCoordinates.
    5340: (19.403794, -99.085594),  # Estadio Alfredo Harp Helu
}


@dataclass(frozen=True)
class BatterUsageEntry:
    game_date: date
    started: bool
    lineup_slot: int | None
    hits: int
    at_bats: int
    plate_appearances: int


@dataclass(frozen=True)
class HitterPlayEntry:
    game_date: date
    game_pk: int | None
    at_bat_number: int
    is_hit: int
    is_at_bat: int
    is_plate_appearance: int = 1


@dataclass(frozen=True)
class PitcherGameEntry:
    game_date: date
    game_pk: int | None
    innings: float
    hits: int
    strikeouts: int
    walks: int
    games_started: int
    games_pitched: int


class MLBClient:
    def __init__(self, *, sleep_seconds: float = 0.03, timeout: int = 30):
        self.session = requests.Session()
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        endpoint = endpoint.lstrip("/")
        response = self.session.get(
            f"{MLB_API_BASE}/{endpoint}",
            params=params or {},
            headers=HEADERS,
            timeout=self.timeout,
        )
        response.raise_for_status()
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return response.json()

    def schedule(self, start: date, end: date, *, hydrate: str = "probablePitcher,venue") -> list[dict]:
        data = self.get(
            "schedule",
            {
                "sportId": 1,
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "hydrate": hydrate,
            },
        )
        games: list[dict] = []
        for day_block in data.get("dates", []) if isinstance(data, dict) else []:
            games.extend(day_block.get("games", []) or [])
        return games

    def teams(self, season: int) -> list[dict]:
        data = self.get("teams", {"sportId": 1, "season": season})
        return data.get("teams", []) if isinstance(data, dict) else []

    def roster(self, team_id: int, *, roster_type: str = "fullSeason", season: int | None = None) -> list[dict]:
        params = {"rosterType": roster_type}
        if season is not None:
            params["season"] = season
        data = self.get(f"teams/{team_id}/roster", params)
        return data.get("roster", []) if isinstance(data, dict) else []

    def people(self, person_ids, *, hydrate: str | None = None, batch_size: int = 50) -> dict[int, dict]:
        ids = []
        seen = set()
        for person_id in person_ids:
            parsed = parse_int(person_id)
            if parsed is None or parsed in seen:
                continue
            seen.add(parsed)
            ids.append(parsed)
        output: dict[int, dict] = {}
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            params = {"personIds": ",".join(str(person_id) for person_id in batch)}
            if hydrate:
                params["hydrate"] = hydrate
            data = self.get("people", params)
            for person in data.get("people", []) if isinstance(data, dict) else []:
                person_id = parse_int(person.get("id"))
                if person_id is not None:
                    output[person_id] = person
        return output

    def boxscore(self, game_pk: int) -> dict:
        return self.get(f"game/{game_pk}/boxscore")

    def team_stats(self, team_id: int, *, group: str, season: int) -> dict:
        return self.get(f"teams/{team_id}/stats", {"stats": "season", "group": group, "season": season})

    def venue(self, venue_id: int, *, hydrate: str | None = "location,timezone") -> dict:
        params = {"hydrate": hydrate} if hydrate else None
        return self.get(f"venues/{venue_id}", params)


def parse_game_datetime(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def get_games_for_date(client: MLBClient, target_date: date) -> list[GameContext]:
    games = []
    for raw in client.schedule(target_date, target_date):
        teams = raw.get("teams", {}) or {}
        away = teams.get("away", {}) or {}
        home = teams.get("home", {}) or {}
        away_team = away.get("team", {}) or {}
        home_team = home.get("team", {}) or {}
        away_id = parse_int(away_team.get("id"))
        home_id = parse_int(home_team.get("id"))
        if away_id is None or home_id is None:
            continue
        away_prob = away.get("probablePitcher", {}) or {}
        home_prob = home.get("probablePitcher", {}) or {}
        venue = raw.get("venue", {}) or {}
        venue_id = parse_int(venue.get("id"))
        coordinates = venue.get("location", {}).get("defaultCoordinates", {}) if isinstance(venue.get("location"), dict) else {}
        if not coordinates and venue_id is not None:
            try:
                venue_payload = client.venue(venue_id)
                venues = venue_payload.get("venues", []) if isinstance(venue_payload, dict) else []
                venue_detail = venues[0] if venues else {}
                coordinates = ((venue_detail.get("location") or {}).get("defaultCoordinates") or {})
            except Exception:
                coordinates = {}
        venue_latitude = parse_float(coordinates.get("latitude"))
        venue_longitude = parse_float(coordinates.get("longitude"))
        if (venue_latitude is None or venue_longitude is None) and venue_id in VENUE_COORDINATE_OVERRIDES:
            venue_latitude, venue_longitude = VENUE_COORDINATE_OVERRIDES[venue_id]
        # StatsAPI often omits venue coordinates on the schedule payload; venue hydration can add them when available.
        games.append(
            GameContext(
                game_pk=parse_int(raw.get("gamePk")),
                game_date=target_date,
                game_datetime_utc=parse_game_datetime(raw.get("gameDate")),
                away_id=away_id,
                home_id=home_id,
                away_abbr=team_abbr(away_id, "AWY"),
                home_abbr=team_abbr(home_id, "HME"),
                away_probable_pitcher_id=parse_int(away_prob.get("id")),
                away_probable_pitcher_name=away_prob.get("fullName") or "TBD",
                home_probable_pitcher_id=parse_int(home_prob.get("id")),
                home_probable_pitcher_name=home_prob.get("fullName") or "TBD",
                venue_id=venue_id,
                venue_name=venue.get("name") or "",
                venue_latitude=venue_latitude,
                venue_longitude=venue_longitude,
            )
        )
    return games


def season_start_for(client: MLBClient, year: int) -> date:
    fallback = date(year, 3, 20)
    try:
        games = client.schedule(date(year, 3, 1), date(year, 4, 30), hydrate="")
    except Exception:
        return fallback
    regular = [game for game in games if game.get("gameType") == "R"]
    if not regular:
        return fallback
    first = canonical_date(regular[0].get("officialDate") or regular[0].get("gameDate"))
    return first or fallback


def build_team_metadata(client: MLBClient, season: int) -> dict[int, dict]:
    output = {}
    for team in client.teams(season):
        team_id = parse_int(team.get("id"))
        if team_id is None:
            continue
        division = team.get("division", {}) or {}
        output[team_id] = {
            "abbr": team.get("abbreviation") or team_abbr(team_id, ""),
            "division_id": parse_int(division.get("id")),
            "name": team.get("name") or "",
        }
    return output


def batting_plate_appearances(stat: dict) -> int:
    direct = parse_int(stat.get("plateAppearances"))
    if direct is not None:
        return direct
    return sum(
        parse_int(stat.get(key)) or 0
        for key in (
            "atBats",
            "baseOnBalls",
            "hitByPitch",
            "sacBunts",
            "sacFlies",
            "catchersInterference",
            "catcherInterference",
        )
    )


def scan_recent_usage_range(
    client: MLBClient,
    start_day: date,
    end_day: date,
    *,
    player_names: dict[int, str],
    player_team_id: dict[int, int],
    logs: dict[int, list[BatterUsageEntry]],
):
    if end_day < start_day:
        return
    games = sorted(
        client.schedule(start_day, end_day, hydrate=""),
        key=lambda game: (game.get("officialDate") or game.get("gameDate") or "", parse_int(game.get("gamePk")) or 0),
    )
    for game in games:
        game_pk = parse_int(game.get("gamePk"))
        current_day = canonical_date(game.get("officialDate") or game.get("gameDate"))
        if game_pk is None or current_day is None:
            continue
        try:
            box = client.boxscore(game_pk)
        except Exception:
            continue
        for side in ("away", "home"):
            side_box = ((box.get("teams") or {}).get(side) or {})
            team_id = parse_int(((side_box.get("team") or {}).get("id")))
            batting_order = [parse_int(pid) for pid in (side_box.get("battingOrder") or [])]
            batting_order = [pid for pid in batting_order if pid is not None]
            starter_slot = {pid: idx for idx, pid in enumerate(batting_order[:9], start=1)}
            for raw_pid in side_box.get("batters", []) or []:
                pid = parse_int(raw_pid)
                if pid is None:
                    continue
                player = (side_box.get("players") or {}).get(f"ID{pid}", {})
                person = player.get("person", {}) or {}
                if person.get("fullName"):
                    player_names[pid] = person.get("fullName")
                if team_id is not None:
                    player_team_id[pid] = team_id
                batting = ((player.get("stats") or {}).get("batting") or {})
                hits = parse_int(batting.get("hits")) or 0
                at_bats = parse_int(batting.get("atBats")) or 0
                plate_appearances = batting_plate_appearances(batting)
                if at_bats == 0 and plate_appearances == 0:
                    continue
                logs[pid].append(
                    BatterUsageEntry(
                        game_date=current_day,
                        started=pid in starter_slot,
                        lineup_slot=starter_slot.get(pid),
                        hits=hits,
                        at_bats=at_bats,
                        plate_appearances=plate_appearances,
                    )
                )


def load_recent_usage(client: MLBClient, season_start: date, end: date, season: int):
    cache_file = cache_path("recent_usage", f"{season}.pkl")
    cached = load_pickle(cache_file)
    logs: dict[int, list[BatterUsageEntry]] = defaultdict(list)
    player_names: dict[int, str] = {}
    player_team_id: dict[int, int] = {}
    last_date = None
    scan_start = season_start
    if isinstance(cached, dict) and cached.get("schema_version") == RECENT_USAGE_CACHE_SCHEMA:
        player_names.update(cached.get("player_names") or {})
        player_team_id.update(cached.get("player_team_id") or {})
        for raw_pid, entries in (cached.get("logs") or {}).items():
            pid = parse_int(raw_pid)
            if pid is not None:
                logs[pid] = list(entries or [])
        last_date = cached.get("last_date")
        if isinstance(last_date, date):
            if last_date >= end:
                trimmed = {
                    pid: [entry for entry in entries if entry.game_date <= end]
                    for pid, entries in logs.items()
                }
                return player_names, player_team_id, trimmed
            scan_start = last_date + timedelta(days=1)
    if scan_start <= end:
        scan_recent_usage_range(
            client,
            scan_start,
            end,
            player_names=player_names,
            player_team_id=player_team_id,
            logs=logs,
        )
        save_pickle(
            cache_file,
            {
                "schema_version": RECENT_USAGE_CACHE_SCHEMA,
                "last_date": end,
                "player_names": player_names,
                "player_team_id": player_team_id,
                "logs": dict(logs),
            },
        )
    return player_names, player_team_id, dict(logs)


def parse_hitter_play_entries(person: dict) -> list[HitterPlayEntry]:
    entries: list[HitterPlayEntry] = []
    for stat_block in person.get("stats", []) or []:
        for split in stat_block.get("splits", []) or []:
            stat = split.get("stat", {}) or {}
            play = stat.get("play", {}) or {}
            details = play.get("details", {}) or {}
            if not details.get("isPlateAppearance"):
                continue
            game_date = canonical_date(split.get("date"))
            if game_date is None:
                continue
            entries.append(
                HitterPlayEntry(
                    game_date=game_date,
                    game_pk=parse_int((split.get("game") or {}).get("gamePk")),
                    at_bat_number=parse_int(play.get("atBatNumber")) or 0,
                    is_hit=1 if details.get("isBaseHit") else 0,
                    is_at_bat=1 if details.get("isAtBat") else 0,
                )
            )
    entries.sort(key=lambda item: (item.game_date, item.game_pk or 0, item.at_bat_number))
    return entries


def load_hitter_play_logs(
    client: MLBClient,
    person_ids,
    *,
    target_date: date,
    seasons_back: int = 20,
    minimum_entries: int = 2500,
) -> dict[int, list[HitterPlayEntry]]:
    player_ids = {pid for pid in (parse_int(value) for value in person_ids) if pid is not None}
    output = {pid: [] for pid in player_ids}
    active = set(player_ids)
    for season in range(target_date.year, max(1900, target_date.year - seasons_back) - 1, -1):
        if not active:
            break
        cache_file = cache_path("hitter_play_logs", f"{season}.pkl")
        cached = load_pickle(cache_file)
        cached_players = {}
        last_refresh_day = None
        if isinstance(cached, dict) and cached.get("schema_version") == PLAY_LOG_CACHE_SCHEMA:
            cached_players = dict(cached.get("players") or {})
            last_refresh_day = cached.get("last_refresh_day")
        refresh_all = season == target_date.year and last_refresh_day != target_date
        ids_to_fetch = active if refresh_all else {pid for pid in active if pid not in cached_players}
        if ids_to_fetch:
            people = client.people(
                ids_to_fetch,
                hydrate=f"stats(group=[hitting],type=[playLog],season={season},gameType=R)",
                batch_size=12,
            )
            for pid in ids_to_fetch:
                cached_players[pid] = parse_hitter_play_entries(people.get(pid, {})) if pid in people else []
            save_pickle(
                cache_file,
                {
                    "schema_version": PLAY_LOG_CACHE_SCHEMA,
                    "last_refresh_day": target_date,
                    "players": cached_players,
                },
            )
        next_active = set()
        for pid in active:
            output[pid].extend(cached_players.get(pid, []))
            prior_entries = [entry for entry in output[pid] if entry.game_date < target_date]
            if len(prior_entries) < minimum_entries:
                next_active.add(pid)
        active = next_active
    for pid in output:
        output[pid].sort(key=lambda item: (item.game_date, item.game_pk or 0, item.at_bat_number))
    return output


def compute_hitter_windows(entries: list[HitterPlayEntry], *, target_date: date) -> dict[str, float | int | None]:
    prior = [entry for entry in entries if entry.game_date < target_date]
    prior.sort(key=lambda item: (item.game_date, item.game_pk or 0, item.at_bat_number), reverse=True)

    def season_ba():
        season_entries = [
            entry for entry in prior
            if entry.game_date.year == target_date.year and entry.is_at_bat
        ]
        hits = sum(entry.is_hit for entry in season_entries)
        at_bats = len(season_entries)
        return (hits / at_bats if at_bats else None), hits, at_bats

    def hipa(pa_window: int):
        used = prior[:pa_window]
        if not used:
            return None, 0, 0
        hits = sum(entry.is_hit for entry in used)
        return hits / len(used), hits, len(used)

    def ba(ab_window: int):
        hits = 0
        at_bats = 0
        for entry in prior:
            if not entry.is_at_bat:
                continue
            hits += entry.is_hit
            at_bats += 1
            if at_bats >= ab_window:
                break
        return (hits / at_bats if at_bats else None), hits, at_bats

    def hipa_through_ab_window(ab_window: int):
        hits = 0
        at_bats = 0
        plate_appearances = 0
        for entry in prior:
            hits += entry.is_hit
            at_bats += entry.is_at_bat
            plate_appearances += 1
            if at_bats >= ab_window:
                break
        return (
            (hits / plate_appearances if plate_appearances else None),
            hits,
            plate_appearances,
            at_bats,
        )

    hipa_2500, hits_2500_pa, sample_2500_pa = hipa(2500)
    hipa_500, hits_500_pa, sample_500_pa = hipa(500)
    hipa_75_ab, hits_75_ab_hipa, sample_75_ab_pa, sample_75_ab_for_hipa = hipa_through_ab_window(75)
    ba_season, hits_season, sample_season = season_ba()
    ba_2500, hits_2500_ab, sample_2500_ab = ba(2500)
    ba_500, hits_500_ab, sample_500_ab = ba(500)
    ba_75, hits_75_ab, sample_75_ab = ba(75)
    ba_25, hits_25_ab, sample_25_ab = ba(25)
    return {
        "hipa_2500_pa": hipa_2500,
        "hipa_2500_hits": hits_2500_pa,
        "hipa_2500_sample": sample_2500_pa,
        "hipa_500_pa": hipa_500,
        "hipa_500_hits": hits_500_pa,
        "hipa_500_sample": sample_500_pa,
        "hipa_75_ab": hipa_75_ab,
        "hipa_75_ab_hits": hits_75_ab_hipa,
        "hipa_75_ab_pa_sample": sample_75_ab_pa,
        "hipa_75_ab_ab_sample": sample_75_ab_for_hipa,
        "ba_season": ba_season,
        "ba_season_hits": hits_season,
        "ba_season_sample": sample_season,
        "ba_2500_ab": ba_2500,
        "ba_2500_hits": hits_2500_ab,
        "ba_2500_sample": sample_2500_ab,
        "ba_500_ab": ba_500,
        "ba_500_hits": hits_500_ab,
        "ba_500_sample": sample_500_ab,
        "ba_75_ab": ba_75,
        "ba_75_hits": hits_75_ab,
        "ba_75_sample": sample_75_ab,
        "ba_25_ab": ba_25,
        "ba_25_hits": hits_25_ab,
        "ba_25_sample": sample_25_ab,
    }


def parse_pitcher_game_entries(person: dict) -> list[PitcherGameEntry]:
    entries: list[PitcherGameEntry] = []
    for stat_block in person.get("stats", []) or []:
        for split in stat_block.get("splits", []) or []:
            game_date = canonical_date(split.get("date"))
            if game_date is None:
                continue
            stat = split.get("stat", {}) or {}
            entries.append(
                PitcherGameEntry(
                    game_date=game_date,
                    game_pk=parse_int((split.get("game") or {}).get("gamePk")),
                    innings=parse_mlb_innings(stat.get("inningsPitched")) or 0.0,
                    hits=parse_int(stat.get("hits")) or 0,
                    strikeouts=parse_int(stat.get("strikeOuts")) or 0,
                    walks=parse_int(stat.get("baseOnBalls")) or 0,
                    games_started=parse_int(stat.get("gamesStarted")) or 0,
                    games_pitched=parse_int(stat.get("gamesPitched")) or 0,
                )
            )
    entries.sort(key=lambda item: (item.game_date, item.game_pk or 0))
    return entries


def load_pitcher_game_logs(
    client: MLBClient,
    person_ids,
    *,
    target_date: date,
    seasons_back: int = 6,
    minimum_innings: float = 350.0,
) -> dict[int, list[PitcherGameEntry]]:
    player_ids = {pid for pid in (parse_int(value) for value in person_ids) if pid is not None}
    output = {pid: [] for pid in player_ids}
    active = set(player_ids)
    for season in range(target_date.year, max(1900, target_date.year - seasons_back) - 1, -1):
        if not active:
            break
        cache_file = cache_path("pitcher_game_logs", f"{season}.pkl")
        cached = load_pickle(cache_file)
        cached_players = {}
        last_refresh_day = None
        if isinstance(cached, dict) and cached.get("schema_version") == PITCHER_LOG_CACHE_SCHEMA:
            cached_players = dict(cached.get("players") or {})
            last_refresh_day = cached.get("last_refresh_day")
        refresh_all = season == target_date.year and last_refresh_day != target_date
        ids_to_fetch = active if refresh_all else {pid for pid in active if pid not in cached_players}
        if ids_to_fetch:
            people = client.people(
                ids_to_fetch,
                hydrate=f"stats(group=[pitching],type=[gameLog],season={season},gameType=R)",
                batch_size=20,
            )
            for pid in ids_to_fetch:
                cached_players[pid] = parse_pitcher_game_entries(people.get(pid, {})) if pid in people else []
            save_pickle(
                cache_file,
                {
                    "schema_version": PITCHER_LOG_CACHE_SCHEMA,
                    "last_refresh_day": target_date,
                    "players": cached_players,
                },
            )
        next_active = set()
        for pid in active:
            output[pid].extend(cached_players.get(pid, []))
            prior_entries = [entry for entry in output[pid] if entry.game_date < target_date]
            if sum(entry.innings for entry in prior_entries) < minimum_innings:
                next_active.add(pid)
        active = next_active
    for pid in output:
        output[pid].sort(key=lambda item: (item.game_date, item.game_pk or 0))
    return output


def pitcher_window_stats(entries: list[PitcherGameEntry], *, target_date: date, innings_window: float) -> dict[str, float | int | date | None]:
    prior = [entry for entry in entries if entry.game_date < target_date and entry.innings > 0]
    prior.sort(key=lambda item: (item.game_date, item.game_pk or 0), reverse=True)
    innings = 0.0
    hits = 0
    cutoff = None
    for entry in prior:
        innings += entry.innings
        hits += entry.hits
        cutoff = entry.game_date
        if innings >= innings_window:
            break
    return {
        "innings": innings,
        "hits": hits,
        "hits_per_inning": safe_divide(hits, innings),
        "cutoff_date": cutoff,
    }


def pitcher_last_start_stats(entries: list[PitcherGameEntry], *, target_date: date) -> dict[str, float | int | date | None]:
    prior_starts = [
        entry for entry in entries
        if entry.game_date < target_date and entry.games_started > 0
    ]
    if not prior_starts:
        return {
            "date": None,
            "innings": None,
            "hits": None,
            "strikeouts": None,
            "walks": None,
        }
    last_start = max(prior_starts, key=lambda item: (item.game_date, item.game_pk or 0))
    return {
        "date": last_start.game_date,
        "innings": last_start.innings,
        "hits": last_start.hits,
        "strikeouts": last_start.strikeouts,
        "walks": last_start.walks,
    }


def load_pitcher_season_context(client: MLBClient, person_ids, season: int) -> dict[int, dict]:
    people = client.people(person_ids, hydrate=f"stats(group=[pitching],type=[season],season={season})")
    output = {}
    for pid, person in people.items():
        pitch_hand = ((person.get("pitchHand") or {}).get("code") or "?").upper()
        context = {
            "pitch_hand": pitch_hand,
            "hits": None,
            "innings": None,
            "hits_per_inning": None,
            "games_started": None,
            "games_pitched": None,
            "avg": None,
        }
        for stat_block in person.get("stats", []) or []:
            for split in stat_block.get("splits", []) or []:
                stat = split.get("stat", {}) or {}
                hits = parse_int(stat.get("hits"))
                innings = parse_mlb_innings(stat.get("inningsPitched"))
                context.update(
                    {
                        "hits": hits,
                        "innings": innings,
                        "hits_per_inning": safe_divide(hits, innings),
                        "games_started": parse_int(stat.get("gamesStarted")),
                        "games_pitched": parse_int(stat.get("gamesPitched")),
                        "avg": parse_float(stat.get("avg") or stat.get("opponentsAverage")),
                    }
                )
                break
            if context["innings"] is not None:
                break
        output[pid] = context
    return output


def likely_opener(pitcher_context: dict, game_logs: list[PitcherGameEntry], *, target_date: date) -> bool:
    games_started = parse_int(pitcher_context.get("games_started")) or 0
    games_pitched = parse_int(pitcher_context.get("games_pitched")) or 0
    innings = parse_float(pitcher_context.get("innings")) or 0.0
    if games_pitched >= 3 and games_started == 0:
        return True
    recent_starts = [
        entry for entry in game_logs
        if entry.game_date < target_date and entry.games_started > 0
    ][-3:]
    if recent_starts and sum(entry.innings for entry in recent_starts) / len(recent_starts) < 3.0:
        return True
    if games_started > 0 and innings / max(games_started, 1) < 3.0 and games_started >= 2:
        return True
    return False


def compute_bullpen_stats(client: MLBClient, season_start: date, end: date, season: int) -> dict[int, dict[str, float | None]]:
    cache_file = cache_path("bullpen_stats", f"{season}.pkl")
    cached = load_pickle(cache_file)
    if (
        isinstance(cached, dict)
        and cached.get("schema_version") == BULLPEN_STATS_CACHE_SCHEMA
        and cached.get("last_date") == end
    ):
        return cached.get("team_stats") or {}

    team_totals = defaultdict(lambda: {"innings": 0.0, "hits": 0, "at_bats": 0})
    for game in client.schedule(season_start, end, hydrate=""):
        status = game.get("status") or {}
        if status.get("startTimeTBD") and parse_int(game.get("gameNumber")) and parse_int(game.get("gameNumber")) > 1:
            # MLB's official team reliever splits exclude these floating-time
            # doubleheader placeholders, even when a boxscore feed exists.
            continue
        game_pk = parse_int(game.get("gamePk"))
        if game_pk is None:
            continue
        try:
            box = client.boxscore(game_pk)
        except Exception:
            continue
        for side in ("away", "home"):
            side_box = ((box.get("teams") or {}).get(side) or {})
            team_id = parse_int(((side_box.get("team") or {}).get("id")))
            pitcher_ids = [parse_int(pid) for pid in (side_box.get("pitchers") or [])]
            pitcher_ids = [pid for pid in pitcher_ids if pid is not None]
            if team_id is None or len(pitcher_ids) <= 1:
                continue
            for pitcher_id in pitcher_ids[1:]:
                player = (side_box.get("players") or {}).get(f"ID{pitcher_id}", {})
                pitching = ((player.get("stats") or {}).get("pitching") or {})
                innings = parse_mlb_innings(pitching.get("inningsPitched")) or 0.0
                hits = parse_int(pitching.get("hits")) or 0
                at_bats = parse_int(pitching.get("atBats")) or 0
                team_totals[team_id]["innings"] += innings
                team_totals[team_id]["hits"] += hits
                team_totals[team_id]["at_bats"] += at_bats
    team_stats = {
        team_id: {
            "hits_per_inning": safe_divide(values["hits"], values["innings"]),
            "opponent_batting_average": safe_divide(values["hits"], values["at_bats"]),
        }
        for team_id, values in team_totals.items()
    }
    save_pickle(
        cache_file,
        {"schema_version": BULLPEN_STATS_CACHE_SCHEMA, "last_date": end, "team_stats": team_stats},
    )
    return team_stats


def compute_bullpen_hpi(client: MLBClient, season_start: date, end: date, season: int) -> dict[int, float | None]:
    team_stats = compute_bullpen_stats(client, season_start, end, season)
    team_hpi = {
        team_id: values.get("hits_per_inning")
        for team_id, values in team_stats.items()
    }
    return team_hpi
