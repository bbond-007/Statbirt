from __future__ import annotations

from collections import defaultdict
import csv
from datetime import date, timedelta, timezone
from pathlib import Path

from .config import PipelineConfig
from .fangraphs import build_stuff_plus_lookup, lookup_stuff_plus
from .lineups import fetch_confirmed_lineups
from .mlb_api import (
    MLBClient,
    build_team_metadata,
    compute_bullpen_stats,
    compute_hitter_windows,
    get_games_for_date,
    likely_opener,
    load_hitter_play_logs,
    load_pitcher_game_logs,
    load_pitcher_season_context,
    load_recent_usage,
    load_roster_availability,
    pitcher_last_start_stats,
    pitcher_window_stats,
    season_start_for,
)
from .models import CandidateFeatures, ScoredCandidate
from .results import RESULT_FIELDS, upsert_candidate_rows
from .savant import (
    SeasonWindow,
    STATCAST_CAREER_START_DATE,
    expected_pa_from_lineup_slot,
    load_or_build_statcast_store,
    load_park_factors,
    load_sprint_speeds,
    projected_bat_side,
    select_park_hit_factor,
)
from .scoring import score_candidate
from .savant_snapshots import load_bat_tracking_snapshot, load_oaa_snapshot
from .utils import format_float, normalize_name, parse_float, parse_int, safe_divide, team_abbr
from .weather import fetch_weather_forecast


def format_datetime_utc(value) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def recent_usage_summary(entries, *, target_date: date, recent_games: int) -> dict:
    prior = [entry for entry in entries if entry.game_date < target_date]
    prior.sort(key=lambda entry: entry.game_date)
    recent = prior[-recent_games:]
    starts = sum(1 for entry in recent if entry.started)
    slots = [entry.lineup_slot for entry in recent if entry.started and entry.lineup_slot is not None]
    return {
        "starts": starts,
        "lineup_slot": (sum(slots) / len(slots)) if slots else None,
    }


def season_pa_per_game(entries, *, target_date: date) -> float | None:
    prior = [entry for entry in entries if entry.game_date < target_date and entry.plate_appearances > 0]
    if not prior:
        return None
    return sum(entry.plate_appearances for entry in prior) / len(prior)


def recent_batting_summary(entries, *, target_date: date, games: int = 5) -> dict[str, float | int | None]:
    prior = [
        entry
        for entry in entries
        if entry.game_date < target_date and (entry.at_bats > 0 or entry.plate_appearances > 0)
    ]
    prior.sort(key=lambda entry: entry.game_date)
    recent = prior[-games:]
    if not recent:
        return {"games": 0, "hits": None, "at_bats": None, "batting_average": None}
    hits = sum(entry.hits for entry in recent)
    at_bats = sum(entry.at_bats for entry in recent)
    return {
        "games": len(recent),
        "hits": hits,
        "at_bats": at_bats,
        "batting_average": safe_divide(hits, at_bats),
    }


def _candidate_ids_for_team(
    *,
    team_id: int,
    confirmed_lineups: dict[int, dict],
    team_hitters: dict[int, list[int]],
    usage_logs: dict[int, list],
    target_date: date,
    config: PipelineConfig,
) -> set[int]:
    lineup = confirmed_lineups.get(team_id, {})
    if lineup.get("confirmed"):
        return {parse_int(pid) for pid in (lineup.get("players_by_id") or {}).keys() if parse_int(pid) is not None}
    output = set()
    for player_id in team_hitters.get(team_id, []):
        summary = recent_usage_summary(
            usage_logs.get(player_id, []),
            target_date=target_date,
            recent_games=config.recent_usage_games,
        )
        if summary["starts"] >= config.min_starts_last_5:
            output.add(player_id)
    return output


def _lineup_slot_for(
    *,
    team_id: int,
    player_id: int,
    player_name: str,
    confirmed_lineups: dict[int, dict],
    usage_logs: dict[int, list],
    target_date: date,
    config: PipelineConfig,
) -> tuple[bool, float | None, int, str, str]:
    lineup = confirmed_lineups.get(team_id, {})
    if lineup.get("confirmed"):
        slot = (lineup.get("players_by_id") or {}).get(player_id)
        if slot is None:
            slot = (lineup.get("players_by_name") or {}).get(normalize_name(player_name))
        return (
            True,
            float(slot) if slot is not None else None,
            0,
            str(lineup.get("source") or "official"),
            str(lineup.get("observed_at_utc") or ""),
        )
    summary = recent_usage_summary(
        usage_logs.get(player_id, []),
        target_date=target_date,
        recent_games=config.recent_usage_games,
    )
    slot = summary.get("lineup_slot")
    return False, float(slot) if slot is not None else None, int(summary.get("starts") or 0), "recent_usage", ""


def _build_statcast_windows(target_date: date, season_start: date, years: int) -> list[SeasonWindow]:
    yesterday = target_date - timedelta(days=1)
    windows: list[SeasonWindow] = []
    if yesterday >= season_start:
        windows.append(
            SeasonWindow(
                season=target_date.year,
                start_date=season_start,
                end_date=yesterday,
                weight=1.0,
            )
        )
    for offset in range(1, max(1, years)):
        season = target_date.year - offset
        windows.append(
            SeasonWindow(
                season=season,
                start_date=date(season, 3, 1),
                end_date=date(season, 11, 30),
                weight=max(0.45, 1.0 - (0.18 * offset)),
            )
        )
    return windows


def _bats_by_player(client: MLBClient, player_ids: set[int]) -> dict[int, dict]:
    people = client.people(player_ids)
    output = {}
    for player_id, person in people.items():
        output[player_id] = {
            "name": person.get("fullName") or person.get("nameFirstLast") or f"Player {player_id}",
            "bats": ((person.get("batSide") or {}).get("code") or "?").upper(),
        }
    return output


def build_daily_candidates(
    target_date: date,
    *,
    config: PipelineConfig | None = None,
    skip_savant: bool = False,
    skip_weather: bool = False,
    confirmed_lineups_override: dict[int, dict] | None = None,
    verbose: bool = False,
) -> tuple[list[ScoredCandidate], list[str]]:
    config = config or PipelineConfig()
    warnings: list[str] = []
    client = MLBClient(sleep_seconds=config.request_sleep_seconds)
    season = target_date.year
    season_start = season_start_for(client, season)
    yesterday = target_date - timedelta(days=1)

    if verbose:
        print(f"Loading MLB schedule for {target_date.isoformat()}...", flush=True)
    games = get_games_for_date(client, target_date)
    if not games:
        return [], ["No MLB games found for target date"]

    team_metadata = build_team_metadata(client, season)
    bat_tracking = {}
    oaa_by_team = {}
    source_metadata = []
    try:
        bat_tracking, bat_metadata = load_bat_tracking_snapshot(target_date)
        if bat_metadata.get("snapshot_id"):
            source_metadata.append(bat_metadata)
    except Exception as exc:
        warnings.append(f"Savant bat-tracking snapshot failed: {type(exc).__name__}: {exc}")
    try:
        oaa_by_team, oaa_metadata = load_oaa_snapshot(target_date, team_metadata)
        if oaa_metadata.get("snapshot_id"):
            source_metadata.append(oaa_metadata)
    except Exception as exc:
        warnings.append(f"Savant OAA snapshot failed: {type(exc).__name__}: {exc}")
    team_game_counts = defaultdict(int)
    for game in games:
        team_game_counts[game.away_id] += 1
        team_game_counts[game.home_id] += 1
    playing_team_ids = {team_id for game in games for team_id in (game.away_id, game.home_id)}
    roster_availability, roster_covered_teams = load_roster_availability(
        client,
        playing_team_ids,
        target_date=target_date,
    )

    if verbose:
        print("Fetching confirmed lineups...", flush=True)
    confirmed_lineups = confirmed_lineups_override if confirmed_lineups_override is not None else fetch_confirmed_lineups()

    if verbose:
        print(f"Building recent hitter usage through {yesterday.isoformat()}...", flush=True)
    player_names, player_team_id, usage_logs = load_recent_usage(client, season_start, yesterday, season)
    team_hitters: dict[int, list[int]] = defaultdict(list)
    for player_id, team_id in player_team_id.items():
        team_hitters[team_id].append(player_id)

    candidate_ids: set[int] = set()
    for game in games:
        for team_id in (game.away_id, game.home_id):
            candidate_ids.update(
                _candidate_ids_for_team(
                    team_id=team_id,
                    confirmed_lineups=confirmed_lineups,
                    team_hitters=team_hitters,
                    usage_logs=usage_logs,
                    target_date=target_date,
                    config=config,
                )
            )
    if not candidate_ids:
        return [], ["No hitter candidates found after recent-start/lineup filtering"]

    pitcher_ids = {
        pitcher_id
        for game in games
        for pitcher_id in (game.away_probable_pitcher_id, game.home_probable_pitcher_id)
        if pitcher_id is not None
    }

    if verbose:
        print(f"Loading player context for {len(candidate_ids)} hitters and {len(pitcher_ids)} probable starters...", flush=True)
    hitter_people = _bats_by_player(client, candidate_ids)
    for player_id, info in hitter_people.items():
        player_names.setdefault(player_id, info["name"])

    hitter_play_logs = load_hitter_play_logs(
        client,
        candidate_ids,
        target_date=target_date,
        seasons_back=config.hitter_play_log_seasons_back,
    )
    hitter_windows = {
        player_id: compute_hitter_windows(entries, target_date=target_date)
        for player_id, entries in hitter_play_logs.items()
    }
    pitcher_context = load_pitcher_season_context(client, pitcher_ids, season)
    pitcher_game_logs = load_pitcher_game_logs(
        client,
        pitcher_ids,
        target_date=target_date,
        seasons_back=config.pitcher_game_log_seasons_back,
    )
    pitcher_windows = {}
    pitcher_last_starts = {}
    for pitcher_id, logs in pitcher_game_logs.items():
        pitcher_windows[pitcher_id] = {
            18: pitcher_window_stats(logs, target_date=target_date, innings_window=18.0),
            50: pitcher_window_stats(logs, target_date=target_date, innings_window=50.0),
            200: pitcher_window_stats(logs, target_date=target_date, innings_window=200.0),
            350: pitcher_window_stats(logs, target_date=target_date, innings_window=350.0),
        }
        pitcher_last_starts[pitcher_id] = pitcher_last_start_stats(logs, target_date=target_date)

    if verbose:
        print("Loading Stuff+, sprint speed, bullpen, and park context...", flush=True)
    stuff_by_id, stuff_by_name, stuff_warnings = build_stuff_plus_lookup(
        season=season,
        manual_csv=config.stuff_plus_csv,
        use_fetch=config.use_fangraphs_fetch,
    )
    warnings.extend(stuff_warnings)
    sprint_speeds = load_sprint_speeds(season)
    park_factors = load_park_factors(season)
    if config.compute_bullpen and yesterday >= season_start:
        bullpen_stats = compute_bullpen_stats(client, season_start, yesterday, season)
    else:
        bullpen_stats = {}
        if not config.compute_bullpen:
            warnings.append("Bullpen computation skipped")

    statcast_store = None
    if skip_savant:
        warnings.append("Savant matchup/split fetch skipped")
    else:
        if verbose:
            print("Building/loading Baseball Savant matchup and split store...", flush=True)
        windows = _build_statcast_windows(target_date, season_start, config.savant_years)
        try:
            statcast_store, cache_hit, cache_file = load_or_build_statcast_store(
                batter_ids=candidate_ids,
                pitcher_ids=pitcher_ids,
                windows=windows,
                h2h_start_date=STATCAST_CAREER_START_DATE,
                h2h_end_date=yesterday,
            )
            if verbose:
                verb = "Loaded cached" if cache_hit else "Built"
                print(f"{verb} Savant store: {cache_file}", flush=True)
        except Exception as exc:
            warnings.append(f"Savant matchup/split fetch failed: {type(exc).__name__}: {exc}")
            statcast_store = None

    weather_by_game = {}
    if config.use_weather and not skip_weather:
        if verbose:
            print("Fetching weather precipitation probabilities...", flush=True)
        for game in games:
            forecast, warning = fetch_weather_forecast(
                latitude=game.venue_latitude,
                longitude=game.venue_longitude,
                game_datetime_utc=game.game_datetime_utc,
            )
            weather_by_game[game.game_pk] = forecast
            if warning:
                warnings.append(f"{game.away_abbr}@{game.home_abbr}: {warning}")

    scored: list[ScoredCandidate] = []
    for game in games:
        contexts = [
            {
                "team_id": game.away_id,
                "team": game.away_abbr,
                "opponent_id": game.home_id,
                "opponent": game.home_abbr,
                "is_home": False,
                "pitcher_id": game.home_probable_pitcher_id,
                "pitcher_name": game.home_probable_pitcher_name,
                "home_team_id": game.home_id,
            },
            {
                "team_id": game.home_id,
                "team": game.home_abbr,
                "opponent_id": game.away_id,
                "opponent": game.away_abbr,
                "is_home": True,
                "pitcher_id": game.away_probable_pitcher_id,
                "pitcher_name": game.away_probable_pitcher_name,
                "home_team_id": game.home_id,
            },
        ]
        for context in contexts:
            team_id = context["team_id"]
            team_candidates = _candidate_ids_for_team(
                team_id=team_id,
                confirmed_lineups=confirmed_lineups,
                team_hitters=team_hitters,
                usage_logs=usage_logs,
                target_date=target_date,
                config=config,
            )
            for player_id in team_candidates:
                player_name = player_names.get(player_id) or hitter_people.get(player_id, {}).get("name") or f"Player {player_id}"
                confirmed, lineup_slot, starts_last_5, lineup_source, lineup_observed_at = _lineup_slot_for(
                    team_id=team_id,
                    player_id=player_id,
                    player_name=player_name,
                    confirmed_lineups=confirmed_lineups,
                    usage_logs=usage_logs,
                    target_date=target_date,
                    config=config,
                )
                if confirmed and lineup_slot is None:
                    continue
                if not confirmed and starts_last_5 < config.min_starts_last_5:
                    continue

                pitcher_id = parse_int(context["pitcher_id"])
                pitcher_name = context["pitcher_name"] or "TBD"
                pctx = pitcher_context.get(pitcher_id, {}) if pitcher_id is not None else {}
                pitcher_hand = (pctx.get("pitch_hand") or "?").upper()
                bats = hitter_people.get(player_id, {}).get("bats") or "?"
                stand = projected_bat_side(bats, pitcher_hand)
                hwin = hitter_windows.get(player_id, {})
                pwin = pitcher_windows.get(pitcher_id, {}) if pitcher_id is not None else {}
                last_start = pitcher_last_starts.get(pitcher_id, {}) if pitcher_id is not None else {}
                split_windows = statcast_store.hitter_split_windows(player_id, target_date=target_date) if statcast_store else {}
                hitter_discipline = (
                    statcast_store.hitter_plate_discipline(
                        player_id,
                        target_date=target_date,
                        current_season=season,
                    )
                    if statcast_store
                    else {}
                )
                hitter_contact = (
                    statcast_store.hitter_contact_quality(player_id, target_date=target_date)
                    if statcast_store
                    else {}
                )
                pitcher_contact = (
                    statcast_store.pitcher_contact_quality_allowed(pitcher_id, target_date=target_date)
                    if statcast_store
                    else {}
                )
                h2h = statcast_store.h2h(player_id, pitcher_id) if statcast_store else None
                inferred = (
                    statcast_store.inferred_pitch_type(
                        batter_id=player_id,
                        pitcher_id=pitcher_id,
                        pitcher_hand=pitcher_hand,
                        stand=stand,
                        target_date=target_date,
                    )
                    if statcast_store
                    else None
                )
                cutoff_50 = (pwin.get(50) or {}).get("cutoff_date")
                cutoff_200 = (pwin.get(200) or {}).get("cutoff_date")
                pitcher_lr_opp_ba = (
                    statcast_store.pitcher_split_opp_ba(pitcher_id, stand)
                    if statcast_store
                    else None
                )
                pitcher_lr_opp_ba_50 = (
                    statcast_store.pitcher_split_opp_ba(pitcher_id, stand, cutoff_date=cutoff_50)
                    if statcast_store
                    else None
                )
                pitcher_lr_opp_ba_200 = (
                    statcast_store.pitcher_split_opp_ba(pitcher_id, stand, cutoff_date=cutoff_200)
                    if statcast_store
                    else None
                )
                matchup_hand_key = "lhp" if pitcher_hand == "L" else "rhp"
                home_team_id = parse_int(context["home_team_id"])
                park_hit_factor = select_park_hit_factor(park_factors, home_team_id, stand) if home_team_id is not None else None
                missing_data = []
                if pitcher_id is None:
                    missing_data.append("Missing probable starter")
                if statcast_store is None:
                    missing_data.append("Missing Savant matchup/split features")
                weather_forecast = weather_by_game.get(game.game_pk)
                last_5_batting = recent_batting_summary(
                    usage_logs.get(player_id, []),
                    target_date=target_date,
                    games=5,
                )
                bat_tracking_row = bat_tracking.get(player_id, {})
                opponent_oaa = oaa_by_team.get(context["opponent_id"], {})
                roster_row = roster_availability.get(player_id)
                feature_snapshot_ids = [str(item.get("snapshot_id")) for item in source_metadata if item.get("snapshot_id")]
                feature_source_hashes = [str(item.get("source_hash")) for item in source_metadata if item.get("source_hash")]
                source_max_dates = [
                    str(item.get("source_max_game_date"))
                    for item in source_metadata
                    if item.get("source_max_game_date")
                ]

                features = CandidateFeatures(
                    target_date=target_date,
                    player_id=player_id,
                    player_name=player_name,
                    team_id=team_id,
                    team=context["team"],
                    opponent_id=context["opponent_id"],
                    opponent=context["opponent"],
                    game_pk=game.game_pk,
                    is_home=context["is_home"],
                    game_start_time_utc=game.game_datetime_utc,
                    venue_name=game.venue_name,
                    confirmed_lineup=confirmed,
                    lineup_source=lineup_source,
                    lineup_observed_at_utc=lineup_observed_at,
                    candidate_pool_source=(
                        "official_lineup"
                        if lineup_source == "official"
                        else "final_boxscore"
                        if lineup_source == "final_boxscore"
                        else "recent_usage"
                    ),
                    active_roster=(
                        True
                        if roster_row is not None
                        else False
                        if team_id in roster_covered_teams
                        else None
                    ),
                    roster_status_code=str((roster_row or {}).get("roster_status_code") or ""),
                    last_transaction_type_code=str(
                        (roster_row or {}).get("last_transaction_type_code") or ""
                    ),
                    last_transaction_date=(roster_row or {}).get("last_transaction_date"),
                    days_since_activation=(roster_row or {}).get("days_since_activation"),
                    lineup_slot=lineup_slot,
                    starts_last_5=starts_last_5,
                    hitter_last_5_games_played=int(last_5_batting.get("games") or 0),
                    hitter_last_5_games_hits=last_5_batting.get("hits"),
                    hitter_last_5_games_ab=last_5_batting.get("at_bats"),
                    hitter_last_5_games_ba=last_5_batting.get("batting_average"),
                    pitcher_id=pitcher_id,
                    pitcher_name=pitcher_name,
                    pitcher_hand=pitcher_hand,
                    batter_stand=stand,
                    same_division=(
                        team_metadata.get(team_id, {}).get("division_id") is not None
                        and team_metadata.get(team_id, {}).get("division_id")
                        == team_metadata.get(context["opponent_id"], {}).get("division_id")
                    ),
                    doubleheader=team_game_counts[team_id] > 1,
                    precipitation_probability=getattr(weather_forecast, "precipitation_probability", None),
                    forecast_temperature_f=getattr(weather_forecast, "temperature_f", None),
                    opener_risk=likely_opener(pctx, pitcher_game_logs.get(pitcher_id, []), target_date=target_date)
                    if pitcher_id is not None
                    else True,
                    hitter_hipa_2500_pa=hwin.get("hipa_2500_pa"),
                    hitter_pa_per_game_season=season_pa_per_game(
                        usage_logs.get(player_id, []),
                        target_date=target_date,
                    ),
                    hitter_ba_season=hwin.get("ba_season"),
                    hitter_ba_2500_ab=hwin.get("ba_2500_ab"),
                    hitter_hipa_500_pa=hwin.get("hipa_500_pa"),
                    hitter_hipa_75_ab=hwin.get("hipa_75_ab"),
                    hitter_ba_75_ab=hwin.get("ba_75_ab"),
                    hitter_ba_25_ab=hwin.get("ba_25_ab"),
                    hitter_ba_500_ab=hwin.get("ba_500_ab"),
                    hitter_bb_rate_season=hitter_discipline.get("bb_rate_season"),
                    hitter_bb_rate_500_pa=hitter_discipline.get("bb_rate_500_pa"),
                    hitter_whiff_rate_season=hitter_discipline.get("whiff_rate_season"),
                    hitter_whiff_rate_500_pa=hitter_discipline.get("whiff_rate_500_pa"),
                    hitter_k_rate_season=hitter_discipline.get("k_rate_season"),
                    hitter_k_rate_500_pa=hitter_discipline.get("k_rate_500_pa"),
                    hitter_xba_100_pa=hitter_contact.get("xba"),
                    hitter_xba_denominator_100_pa=hitter_contact.get("xba_denominator"),
                    hitter_contact_xba_50_bbe=hitter_contact.get("contact_xba"),
                    hitter_hard_hit_rate_50_bbe=hitter_contact.get("hard_hit_rate"),
                    hitter_sweet_spot_rate_50_bbe=hitter_contact.get("sweet_spot_rate"),
                    hitter_ev50_50_bbe=hitter_contact.get("ev50"),
                    hitter_contact_bbe_50=hitter_contact.get("bbe"),
                    hitter_bat_speed_50_swings=hitter_contact.get("bat_speed"),
                    hitter_swing_length_50_swings=hitter_contact.get("swing_length"),
                    hitter_tracked_swings_50=hitter_contact.get("swings"),
                    hitter_competitive_swings_season=bat_tracking_row.get("competitive_swings"),
                    hitter_competitive_contact_rate_season=bat_tracking_row.get("competitive_contact_rate"),
                    hitter_avg_bat_speed_season=bat_tracking_row.get("avg_bat_speed"),
                    hitter_avg_swing_length_season=bat_tracking_row.get("avg_swing_length"),
                    hitter_squared_up_per_contact_season=bat_tracking_row.get("squared_up_per_contact"),
                    hitter_blast_per_contact_season=bat_tracking_row.get("blast_per_contact"),
                    hitter_split_ba_season_vs_lhp=split_windows.get("ba_season_vs_lhp"),
                    hitter_split_ba_season_vs_rhp=split_windows.get("ba_season_vs_rhp"),
                    hitter_split_pa_season_vs_lhp=split_windows.get("pa_season_vs_lhp"),
                    hitter_split_pa_season_vs_rhp=split_windows.get("pa_season_vs_rhp"),
                    hitter_split_ba_500_vs_lhp=split_windows.get("ba_500_vs_lhp"),
                    hitter_split_ba_500_vs_rhp=split_windows.get("ba_500_vs_rhp"),
                    hitter_split_ba_1500_vs_lhp=split_windows.get("ba_1500_vs_lhp"),
                    hitter_split_ba_1500_vs_rhp=split_windows.get("ba_1500_vs_rhp"),
                    hitter_matchup_hand_ba_500=split_windows.get(f"ba_500_vs_{matchup_hand_key}"),
                    hitter_matchup_hand_ba_1500=split_windows.get(f"ba_1500_vs_{matchup_hand_key}"),
                    pitcher_hpi_350=(pwin.get(350) or {}).get("hits_per_inning"),
                    pitcher_hpi_200=(pwin.get(200) or {}).get("hits_per_inning"),
                    pitcher_hpi_season=pctx.get("hits_per_inning"),
                    pitcher_hits_last_18_ip=(pwin.get(18) or {}).get("hits"),
                    pitcher_last_start_date=last_start.get("date"),
                    pitcher_last_start_ip=last_start.get("innings"),
                    pitcher_last_start_hits=last_start.get("hits"),
                    pitcher_last_start_strikeouts=last_start.get("strikeouts"),
                    pitcher_last_start_walks=last_start.get("walks"),
                    pitcher_stuff_plus=lookup_stuff_plus(
                        player_id=pitcher_id,
                        player_name=pitcher_name,
                        by_id=stuff_by_id,
                        by_name=stuff_by_name,
                    ),
                    pitcher_xba_allowed_100_bf=pitcher_contact.get("xba"),
                    pitcher_xba_denominator_100_bf=pitcher_contact.get("xba_denominator"),
                    pitcher_contact_xba_allowed_50_bbe=pitcher_contact.get("contact_xba"),
                    pitcher_hard_hit_rate_allowed_50_bbe=pitcher_contact.get("hard_hit_rate"),
                    pitcher_sweet_spot_rate_allowed_50_bbe=pitcher_contact.get("sweet_spot_rate"),
                    pitcher_ev50_allowed_50_bbe=pitcher_contact.get("ev50"),
                    pitcher_contact_bbe_50=pitcher_contact.get("bbe"),
                    h2h_pa=h2h.pa if h2h else 0,
                    h2h_hits=h2h.hits if h2h else 0,
                    h2h_hit_rate=h2h.hit_rate if h2h else None,
                    h2h_whiff_rate=h2h.whiff_rate if h2h else None,
                    h2h_k_rate=h2h.k_rate if h2h else None,
                    h2h_exit_velocity=h2h.exit_velocity if h2h else None,
                    h2h_xba=h2h.xba if h2h else None,
                    pitcher_lr_opp_ba=pitcher_lr_opp_ba,
                    pitcher_lr_opp_ba_50=pitcher_lr_opp_ba_50,
                    pitcher_lr_opp_ba_200=pitcher_lr_opp_ba_200,
                    inferred_pitch_type_ba=inferred.ba if inferred else None,
                    inferred_pitch_type_xba=inferred.xba if inferred else None,
                    inferred_pitch_type_contact_rate=inferred.contact_rate if inferred else None,
                    inferred_pitch_type_shape_distance=inferred.shape_distance if inferred else None,
                    inferred_pitch_type_coverage=inferred.coverage if inferred else None,
                    bullpen_hpi=(bullpen_stats.get(context["opponent_id"]) or {}).get("hits_per_inning"),
                    bullpen_opp_ba=(bullpen_stats.get(context["opponent_id"]) or {}).get("opponent_batting_average"),
                    bullpen_pitches_3d=(bullpen_stats.get(context["opponent_id"]) or {}).get("pitches_last_3_days"),
                    bullpen_relief_appearances_3d=(bullpen_stats.get(context["opponent_id"]) or {}).get(
                        "relief_appearances_last_3_days"
                    ),
                    bullpen_relievers_used_3d=(bullpen_stats.get(context["opponent_id"]) or {}).get(
                        "relievers_used_last_3_days"
                    ),
                    bullpen_high_workload_relievers_3d=(bullpen_stats.get(context["opponent_id"]) or {}).get(
                        "high_workload_relievers_last_3_days"
                    ),
                    opponent_team_oaa=opponent_oaa.get("team_oaa"),
                    opponent_infield_oaa=opponent_oaa.get("infield_oaa"),
                    opponent_outfield_oaa=opponent_oaa.get("outfield_oaa"),
                    feature_snapshot_id="|".join(feature_snapshot_ids),
                    feature_source_hash="|".join(feature_source_hashes),
                    feature_source_max_game_date=max(source_max_dates, default=""),
                    sprint_speed=sprint_speeds.get(player_id),
                    park_hit_factor=park_hit_factor,
                    expected_pa=expected_pa_from_lineup_slot(lineup_slot),
                    missing_data=missing_data,
                )
                scored.append(score_candidate(features, config))

    scored.sort(
        key=lambda row: (
            not row.valve_result.pickable,
            -row.score,
            row.features.player_name,
        )
    )
    return scored, warnings


def component_map(candidate: ScoredCandidate) -> dict[str, float]:
    return {component.name: component.subscore for component in candidate.components}


def scored_candidate_to_row(candidate: ScoredCandidate) -> dict[str, str]:
    f = candidate.features
    components = component_map(candidate)
    row = {
        "date": f.target_date.isoformat(),
        "player": f.player_name,
        "player_id": str(f.player_id),
        "team": f.team,
        "team_id": str(f.team_id),
        "opponent": f.opponent,
        "opponent_id": str(f.opponent_id),
        "game_pk": "" if f.game_pk is None else str(f.game_pk),
        "game_start_time_utc": format_datetime_utc(f.game_start_time_utc),
        "venue_name": f.venue_name,
        "pickable": "Y" if candidate.valve_result.pickable else "N",
        "score": format_float(candidate.score, 2),
        "hard_pass_reasons": " | ".join(candidate.valve_result.hard_pass_reasons),
        "concerns": " | ".join(candidate.valve_result.concerns),
        "confirmed_lineup": "Y" if f.confirmed_lineup else "N",
        "lineup_source": f.lineup_source,
        "lineup_observed_at_utc": f.lineup_observed_at_utc,
        "candidate_pool_source": f.candidate_pool_source,
        "active_roster": "" if f.active_roster is None else "Y" if f.active_roster else "N",
        "roster_status_code": f.roster_status_code,
        "last_transaction_type_code": f.last_transaction_type_code,
        "last_transaction_date": f.last_transaction_date.isoformat() if f.last_transaction_date else "",
        "days_since_activation": "" if f.days_since_activation is None else str(f.days_since_activation),
        "lineup_slot": format_float(f.lineup_slot, 1),
        "expected_pa": format_float(f.expected_pa, 2),
        "starts_last_5": str(f.starts_last_5),
        "hitter_last_5_games_played": str(f.hitter_last_5_games_played),
        "hitter_last_5_games_hits": "" if f.hitter_last_5_games_hits is None else str(f.hitter_last_5_games_hits),
        "hitter_last_5_games_ab": "" if f.hitter_last_5_games_ab is None else str(f.hitter_last_5_games_ab),
        "hitter_last_5_games_ba": format_float(f.hitter_last_5_games_ba, 3),
        "road_game": "Y" if not f.is_home else "N",
        "division_matchup": "Y" if f.same_division else "N",
        "doubleheader": "Y" if f.doubleheader else "N",
        "precip_probability": format_float(f.precipitation_probability, 1),
        "forecast_temperature_f": format_float(f.forecast_temperature_f, 1),
        "probable_pitcher": f.pitcher_name,
        "probable_pitcher_id": "" if f.pitcher_id is None else str(f.pitcher_id),
        "pitcher_hand": f.pitcher_hand,
        "batter_stand": f.batter_stand,
        "hitter_hipa_2500_pa": format_float(f.hitter_hipa_2500_pa, 3),
        "hitter_pa_per_game_season": format_float(f.hitter_pa_per_game_season, 2),
        "hitter_ba_season": format_float(f.hitter_ba_season, 3),
        "hitter_ba_2500_ab": format_float(f.hitter_ba_2500_ab, 3),
        "hitter_hipa_500_pa": format_float(f.hitter_hipa_500_pa, 3),
        "hitter_hipa_75_ab": format_float(f.hitter_hipa_75_ab, 3),
        "hitter_ba_75_ab": format_float(f.hitter_ba_75_ab, 3),
        "hitter_ba_25_ab": format_float(f.hitter_ba_25_ab, 3),
        "hitter_ba_500_ab": format_float(f.hitter_ba_500_ab, 3),
        "hitter_bb_rate_season": format_float(f.hitter_bb_rate_season, 3),
        "hitter_bb_rate_500_pa": format_float(f.hitter_bb_rate_500_pa, 3),
        "hitter_whiff_rate_season": format_float(f.hitter_whiff_rate_season, 3),
        "hitter_whiff_rate_500_pa": format_float(f.hitter_whiff_rate_500_pa, 3),
        "hitter_k_rate_season": format_float(f.hitter_k_rate_season, 3),
        "hitter_k_rate_500_pa": format_float(f.hitter_k_rate_500_pa, 3),
        "hitter_xba_100_pa": format_float(f.hitter_xba_100_pa, 3),
        "hitter_xba_denominator_100_pa": (
            "" if f.hitter_xba_denominator_100_pa is None else str(f.hitter_xba_denominator_100_pa)
        ),
        "hitter_contact_xba_50_bbe": format_float(f.hitter_contact_xba_50_bbe, 3),
        "hitter_hard_hit_rate_50_bbe": format_float(f.hitter_hard_hit_rate_50_bbe, 3),
        "hitter_sweet_spot_rate_50_bbe": format_float(f.hitter_sweet_spot_rate_50_bbe, 3),
        "hitter_ev50_50_bbe": format_float(f.hitter_ev50_50_bbe, 1),
        "hitter_contact_bbe_50": "" if f.hitter_contact_bbe_50 is None else str(f.hitter_contact_bbe_50),
        "hitter_bat_speed_50_swings": format_float(f.hitter_bat_speed_50_swings, 1),
        "hitter_swing_length_50_swings": format_float(f.hitter_swing_length_50_swings, 1),
        "hitter_tracked_swings_50": "" if f.hitter_tracked_swings_50 is None else str(f.hitter_tracked_swings_50),
        "hitter_competitive_swings_season": (
            "" if f.hitter_competitive_swings_season is None else str(f.hitter_competitive_swings_season)
        ),
        "hitter_competitive_contact_rate_season": format_float(
            f.hitter_competitive_contact_rate_season, 3
        ),
        "hitter_avg_bat_speed_season": format_float(f.hitter_avg_bat_speed_season, 1),
        "hitter_avg_swing_length_season": format_float(f.hitter_avg_swing_length_season, 1),
        "hitter_squared_up_per_contact_season": format_float(
            f.hitter_squared_up_per_contact_season, 3
        ),
        "hitter_blast_per_contact_season": format_float(f.hitter_blast_per_contact_season, 3),
        "hitter_split_ba_season_vs_lhp": format_float(f.hitter_split_ba_season_vs_lhp, 3),
        "hitter_split_ba_season_vs_rhp": format_float(f.hitter_split_ba_season_vs_rhp, 3),
        "hitter_split_pa_season_vs_lhp": "" if f.hitter_split_pa_season_vs_lhp is None else str(f.hitter_split_pa_season_vs_lhp),
        "hitter_split_pa_season_vs_rhp": "" if f.hitter_split_pa_season_vs_rhp is None else str(f.hitter_split_pa_season_vs_rhp),
        "hitter_split_ba_500_vs_lhp": format_float(f.hitter_split_ba_500_vs_lhp, 3),
        "hitter_split_ba_500_vs_rhp": format_float(f.hitter_split_ba_500_vs_rhp, 3),
        "hitter_split_ba_1500_vs_lhp": format_float(f.hitter_split_ba_1500_vs_lhp, 3),
        "hitter_split_ba_1500_vs_rhp": format_float(f.hitter_split_ba_1500_vs_rhp, 3),
        "pitcher_hpi_350": format_float(f.pitcher_hpi_350, 3),
        "pitcher_hpi_200": format_float(f.pitcher_hpi_200, 3),
        "pitcher_hpi_season": format_float(f.pitcher_hpi_season, 3),
        "pitcher_hits_last_18_ip": "" if f.pitcher_hits_last_18_ip is None else str(f.pitcher_hits_last_18_ip),
        "pitcher_last_start_date": "" if f.pitcher_last_start_date is None else f.pitcher_last_start_date.isoformat(),
        "pitcher_last_start_ip": format_float(f.pitcher_last_start_ip, 1),
        "pitcher_last_start_hits": "" if f.pitcher_last_start_hits is None else str(f.pitcher_last_start_hits),
        "pitcher_last_start_strikeouts": "" if f.pitcher_last_start_strikeouts is None else str(f.pitcher_last_start_strikeouts),
        "pitcher_last_start_walks": "" if f.pitcher_last_start_walks is None else str(f.pitcher_last_start_walks),
        "pitcher_stuff_plus": format_float(f.pitcher_stuff_plus, 1),
        "pitcher_xba_allowed_100_bf": format_float(f.pitcher_xba_allowed_100_bf, 3),
        "pitcher_xba_denominator_100_bf": (
            "" if f.pitcher_xba_denominator_100_bf is None else str(f.pitcher_xba_denominator_100_bf)
        ),
        "pitcher_contact_xba_allowed_50_bbe": format_float(f.pitcher_contact_xba_allowed_50_bbe, 3),
        "pitcher_hard_hit_rate_allowed_50_bbe": format_float(f.pitcher_hard_hit_rate_allowed_50_bbe, 3),
        "pitcher_sweet_spot_rate_allowed_50_bbe": format_float(f.pitcher_sweet_spot_rate_allowed_50_bbe, 3),
        "pitcher_ev50_allowed_50_bbe": format_float(f.pitcher_ev50_allowed_50_bbe, 1),
        "pitcher_contact_bbe_50": "" if f.pitcher_contact_bbe_50 is None else str(f.pitcher_contact_bbe_50),
        "h2h_pa": str(f.h2h_pa),
        "h2h_hits": str(f.h2h_hits),
        "h2h_hit_rate": format_float(f.h2h_hit_rate, 3),
        "h2h_whiff_rate": format_float(f.h2h_whiff_rate, 3),
        "h2h_k_rate": format_float(f.h2h_k_rate, 3),
        "h2h_exit_velocity": format_float(f.h2h_exit_velocity, 1),
        "h2h_xba": format_float(f.h2h_xba, 3),
        "pitcher_lr_opp_ba": format_float(f.pitcher_lr_opp_ba, 3),
        "pitcher_lr_opp_ba_50": format_float(f.pitcher_lr_opp_ba_50, 3),
        "pitcher_lr_opp_ba_200": format_float(f.pitcher_lr_opp_ba_200, 3),
        "inferred_pitch_type_ba": format_float(f.inferred_pitch_type_ba, 3),
        "inferred_pitch_type_xba": format_float(f.inferred_pitch_type_xba, 3),
        "inferred_pitch_type_contact_rate": format_float(f.inferred_pitch_type_contact_rate, 3),
        "inferred_pitch_type_shape_distance": format_float(f.inferred_pitch_type_shape_distance, 3),
        "inferred_pitch_type_coverage": format_float(f.inferred_pitch_type_coverage, 3),
        "bullpen_hpi": format_float(f.bullpen_hpi, 3),
        "bullpen_opp_ba": format_float(f.bullpen_opp_ba, 3),
        "bullpen_pitches_3d": "" if f.bullpen_pitches_3d is None else str(f.bullpen_pitches_3d),
        "bullpen_relief_appearances_3d": (
            "" if f.bullpen_relief_appearances_3d is None else str(f.bullpen_relief_appearances_3d)
        ),
        "bullpen_relievers_used_3d": (
            "" if f.bullpen_relievers_used_3d is None else str(f.bullpen_relievers_used_3d)
        ),
        "bullpen_high_workload_relievers_3d": (
            "" if f.bullpen_high_workload_relievers_3d is None else str(f.bullpen_high_workload_relievers_3d)
        ),
        "opponent_team_oaa": format_float(f.opponent_team_oaa, 1),
        "opponent_infield_oaa": format_float(f.opponent_infield_oaa, 1),
        "opponent_outfield_oaa": format_float(f.opponent_outfield_oaa, 1),
        "feature_snapshot_id": f.feature_snapshot_id,
        "feature_source_hash": f.feature_source_hash,
        "feature_source_max_game_date": f.feature_source_max_game_date,
        "sprint_speed": format_float(f.sprint_speed, 1),
        "park_hit_factor": format_float(f.park_hit_factor, 1),
    }
    for field in RESULT_FIELDS:
        row[field] = ""
    for name, subscore in components.items():
        row[f"component_{name}"] = format_float(subscore, 2)
    return row


def write_candidates_csv(path: str | Path, candidates: list[ScoredCandidate]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [scored_candidate_to_row(candidate) for candidate in candidates]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_candidates_csv(path: str | Path, candidates: list[ScoredCandidate]) -> dict[str, int]:
    rows = [scored_candidate_to_row(candidate) for candidate in candidates]
    if not rows:
        return {"inserted": 0, "updated": 0, "total_rows": 0}
    replace_date = rows[0].get("date") if rows else None
    return upsert_candidate_rows(path, rows, replace_date=replace_date)
