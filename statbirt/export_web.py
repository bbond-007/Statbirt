from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import DEFAULT_OUTPUT_CSV
from .mlb_api import MLBClient, batting_plate_appearances, compute_bullpen_stats, parse_game_datetime, season_start_for
from .utils import normalize_name, parse_float, parse_int

DEFAULT_WEB_DATA_DIR = Path(__file__).resolve().parents[1] / "web" / "data"
DEFAULT_WEB_JSON = DEFAULT_WEB_DATA_DIR / "top_picks.json"
DEFAULT_DASHBOARD_INDEX = DEFAULT_WEB_DATA_DIR / "dashboard_index.json"
DEFAULT_ARCHIVE_DIR = DEFAULT_WEB_DATA_DIR / "dashboards"
DEFAULT_CONGREGATION_CSV = Path(__file__).resolve().parents[1] / "data" / "manual" / "congregation.csv"

COMPONENT_BUCKETS = {
    "hitter": [
        "component_hitter.hipa_2500_pa",
        "component_hitter.pa_per_game_season",
        "component_hitter.hipa_500_pa",
        "component_hitter.hipa_75_ab",
    ],
    "starting_pitcher": [
        "component_starting_pitcher.hpi_350",
        "component_starting_pitcher.hpi_season",
        "component_starting_pitcher.stuff_plus",
    ],
    "h2h": [
        "component_h2h.direct",
        "component_h2h.pitcher_lr_opp_ba",
        "component_h2h.inferred_pitch_type",
    ],
    "bullpen": [
        "component_bullpen.hpi_season",
    ],
    "other": [
        "component_other.road_game",
        "component_other.division_matchup",
        "component_other.sprint_speed",
        "component_other.park_hit_factor",
        "component_other.lineup_opportunity",
    ],
}

COMPONENT_WEIGHTS = {
    "component_hitter.hipa_2500_pa": 10.0,
    "component_hitter.pa_per_game_season": 7.0,
    "component_hitter.hipa_500_pa": 5.0,
    "component_hitter.hipa_75_ab": 3.0,
    "component_starting_pitcher.hpi_350": 10.0,
    "component_starting_pitcher.hpi_season": 5.0,
    "component_starting_pitcher.stuff_plus": 10.0,
    "component_h2h.direct": 10.0,
    "component_h2h.pitcher_lr_opp_ba": 5.0,
    "component_h2h.inferred_pitch_type": 5.0,
    "component_bullpen.hpi_season": 15.0,
    "component_other.road_game": 3.0,
    "component_other.division_matchup": 3.0,
    "component_other.sprint_speed": 3.0,
    "component_other.park_hit_factor": 3.0,
    "component_other.lineup_opportunity": 3.0,
}

BUCKET_LABELS = {
    "hitter": "Hitter",
    "starting_pitcher": "Starter",
    "h2h": "H2H",
    "bullpen": "Bullpen",
    "other": "Other",
}


def load_congregation(path: Path = DEFAULT_CONGREGATION_CSV) -> dict[str, dict]:
    lookup = {"by_id": {}, "by_name": {}}
    if not path.exists():
        return lookup
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            player = (row.get("player") or "").strip()
            status = (row.get("status") or "").strip()
            player_id = parse_int(row.get("player_id"))
            aliases = [player, *str(row.get("aliases") or "").split("|")]
            record = {
                "player_id": "" if player_id is None else str(player_id),
                "player": player,
                "status": status,
            }
            if player_id is not None:
                lookup["by_id"][str(player_id)] = record
            for alias in aliases:
                key = normalize_name(alias)
                if key:
                    lookup["by_name"][key] = record
    return lookup


def congregation_record_for(row: dict[str, str], congregation: dict[str, dict] | None) -> dict | None:
    if not congregation:
        return None
    player_id = str(row.get("player_id") or "").strip()
    if player_id and player_id in congregation.get("by_id", {}):
        return congregation["by_id"][player_id]
    return congregation.get("by_name", {}).get(normalize_name(row.get("player", "")))


def split_reasons(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(" | ") if part.strip()]


def float_value(row: dict[str, str], field: str) -> float | None:
    return parse_float(row.get(field))


def component_points(row: dict[str, str], field: str) -> float:
    subscore = float_value(row, field) or 0.0
    weight = COMPONENT_WEIGHTS.get(field, 0.0)
    return (weight / 100.0) * subscore


def points(row: dict[str, str], fields: list[str]) -> float:
    return round(sum(component_points(row, field) for field in fields), 2)


def format_rate(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def format_percent_rate(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def format_number(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def format_datetime_utc(value: str | None) -> str:
    parsed = parse_game_datetime(value)
    if parsed is None:
        return ""
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


GAME_STATE_LABELS = {
    "not_started": "Not started",
    "postponed": "Postponed",
    "live_no_hit": "No hit yet",
    "hit": "Hit recorded",
    "final_no_hit": "Final: 0 H",
    "unknown": "Status unavailable",
}


def game_status_text(game: dict) -> str:
    status = game.get("status") or {}
    if isinstance(status, dict):
        return status.get("detailedState") or status.get("abstractGameState") or ""
    return str(status or "")


def game_phase(game: dict) -> str:
    status = game.get("status") or {}
    detailed = ""
    abstract = ""
    if isinstance(status, dict):
        detailed = str(status.get("detailedState") or "")
        abstract = str(status.get("abstractGameState") or "")
    text = f"{abstract} {detailed}".lower()
    if any(token in text for token in ("postponed", "cancelled", "canceled")):
        return "postponed"
    if any(token in text for token in ("final", "game over", "completed")):
        return "final"
    if any(token in text for token in ("preview", "pre-game", "scheduled", "warmup", "delayed start")):
        return "not_started"
    return "live"


def game_schedule_info(game: dict) -> dict[str, str]:
    venue = game.get("venue") or {}
    return {
        "venue_name": venue.get("name") or "",
        "game_start_time_utc": format_datetime_utc(game.get("gameDate")),
    }


def player_hits_from_boxscore(boxscore: dict, player_id: int) -> dict:
    for side in ("away", "home"):
        side_box = ((boxscore.get("teams") or {}).get(side) or {})
        players = side_box.get("players") or {}
        player_block = players.get(f"ID{player_id}")
        if player_block is None:
            for key, value in players.items():
                if parse_int(str(key).replace("ID", "")) == player_id:
                    player_block = value
                    break
        if player_block is None:
            continue
        batting = ((player_block.get("stats") or {}).get("batting") or {})
        hits = parse_int(batting.get("hits")) or 0
        at_bats = parse_int(batting.get("atBats")) or 0
        plate_appearances = batting_plate_appearances(batting)
        return {
            "hits": hits,
            "at_bats": at_bats,
            "plate_appearances": plate_appearances,
            "appeared": at_bats > 0 or plate_appearances > 0,
        }
    return {"hits": 0, "at_bats": 0, "plate_appearances": 0, "appeared": False}


def row_state_key(row: dict[str, str]) -> tuple[int, int] | None:
    game_pk = parse_int(row.get("game_pk"))
    player_id = parse_int(row.get("player_id"))
    if game_pk is None or player_id is None:
        return None
    return game_pk, player_id


def build_game_state_lookup(target_day: date | None, rows: list[dict[str, str]]) -> dict[tuple[int, int], dict]:
    if target_day is None or not rows:
        return {}
    keys_by_game: dict[int, set[int]] = {}
    for row in rows:
        key = row_state_key(row)
        if key is None:
            continue
        game_pk, player_id = key
        keys_by_game.setdefault(game_pk, set()).add(player_id)
    if not keys_by_game:
        return {}

    client = MLBClient()
    try:
        games = client.schedule(target_day, target_day, hydrate="")
    except Exception:
        return {}

    output: dict[tuple[int, int], dict] = {}
    for game in games:
        game_pk = parse_int(game.get("gamePk"))
        if game_pk not in keys_by_game:
            continue
        phase = game_phase(game)
        status_text = game_status_text(game)
        schedule_info = game_schedule_info(game)
        if phase in {"not_started", "postponed"}:
            for player_id in keys_by_game[game_pk]:
                output[(game_pk, player_id)] = {
                    "state": phase,
                    "status": status_text,
                    "hits": None,
                    **schedule_info,
                }
            continue
        try:
            boxscore = client.boxscore(game_pk)
        except Exception:
            for player_id in keys_by_game[game_pk]:
                output[(game_pk, player_id)] = {
                    "state": "unknown",
                    "status": status_text,
                    "hits": None,
                    **schedule_info,
                }
            continue
        for player_id in keys_by_game[game_pk]:
            result = player_hits_from_boxscore(boxscore, player_id)
            hits = int(result.get("hits") or 0)
            if hits > 0:
                state = "hit"
            elif phase == "final":
                state = "final_no_hit"
            else:
                state = "live_no_hit"
            output[(game_pk, player_id)] = {
                "state": state,
                "status": status_text,
                "hits": hits,
                "at_bats": result.get("at_bats"),
                "plate_appearances": result.get("plate_appearances"),
                "appeared": result.get("appeared"),
                **schedule_info,
            }
    return output


def build_bullpen_stat_lookup(target_day: date | None, rows: list[dict[str, str]]) -> dict[int, dict[str, float | None]]:
    if target_day is None or not rows:
        return {}
    if all(float_value(row, "bullpen_opp_ba") is not None for row in rows):
        return {}
    if not any(parse_int(row.get("opponent_id")) is not None for row in rows):
        return {}
    client = MLBClient()
    season_start = season_start_for(client, target_day.year)
    end = target_day - timedelta(days=1)
    if end < season_start:
        return {}
    try:
        return compute_bullpen_stats(client, season_start, end, target_day.year)
    except Exception:
        return {}


def bullpen_opp_ba_for(row: dict[str, str], bullpen_stats: dict[int, dict[str, float | None]] | None) -> float | None:
    from_row = float_value(row, "bullpen_opp_ba")
    if from_row is not None:
        return from_row
    opponent_id = parse_int(row.get("opponent_id"))
    if opponent_id is None:
        return None
    return (bullpen_stats or {}).get(opponent_id, {}).get("opponent_batting_average")


def candidate_payload(
    row: dict[str, str],
    rank: int,
    game_states: dict[tuple[int, int], dict] | None = None,
    bullpen_stats: dict[int, dict[str, float | None]] | None = None,
    congregation: dict[str, dict] | None = None,
) -> dict:
    buckets = {
        key: {
            "label": BUCKET_LABELS[key],
            "points": points(row, fields),
        }
        for key, fields in COMPONENT_BUCKETS.items()
    }
    score = float_value(row, "score") or 0.0
    hard_pass_reasons = split_reasons(row.get("hard_pass_reasons"))
    concerns = split_reasons(row.get("concerns"))
    state_key = row_state_key(row)
    game_state = (game_states or {}).get(state_key, {}) if state_key else {}
    state = game_state.get("state") or "unknown"
    bullpen_opp_ba = bullpen_opp_ba_for(row, bullpen_stats)
    venue_name = row.get("venue_name") or game_state.get("venue_name") or ""
    game_start_time_utc = row.get("game_start_time_utc") or game_state.get("game_start_time_utc") or ""
    congregation_record = congregation_record_for(row, congregation)
    return {
        "rank": rank,
        "date": row.get("date") or "",
        "player": row.get("player") or "",
        "player_id": row.get("player_id") or "",
        "team": row.get("team") or "",
        "opponent": row.get("opponent") or "",
        "game_pk": row.get("game_pk") or "",
        "game_start_time_utc": game_start_time_utc,
        "venue_name": venue_name,
        "congregation_status": (congregation_record or {}).get("status", ""),
        "congregation_member": congregation_record is not None,
        "pickable": str(row.get("pickable") or "").upper() == "Y",
        "score": round(score, 2),
        "probable_pitcher": row.get("probable_pitcher") or "TBD",
        "pitcher_hand": row.get("pitcher_hand") or "?",
        "batter_stand": row.get("batter_stand") or "?",
        "lineup_slot": format_number(float_value(row, "lineup_slot"), digits=1),
        "expected_pa": format_number(float_value(row, "expected_pa"), digits=1),
        "confirmed_lineup": row.get("confirmed_lineup") or "",
        "precip_probability": format_number(float_value(row, "precip_probability"), digits=1),
        "game_state": state,
        "game_state_label": GAME_STATE_LABELS.get(state, GAME_STATE_LABELS["unknown"]),
        "game_status": game_state.get("status") or "",
        "game_hits": game_state.get("hits"),
        "bullpen_opp_ba": format_rate(bullpen_opp_ba),
        "hard_pass_reasons": hard_pass_reasons,
        "concerns": concerns,
        "primary_risk": (hard_pass_reasons or concerns or ["Clean"])[0],
        "buckets": buckets,
        "factors": {
            "hitter": [
                {"label": "HiPA 2500", "value": format_rate(float_value(row, "hitter_hipa_2500_pa"))},
                {"label": "PA/G", "value": format_number(float_value(row, "hitter_pa_per_game_season"), digits=2)},
                {"label": "HiPA 500", "value": format_rate(float_value(row, "hitter_hipa_500_pa"))},
                {"label": "HiPA 75 AB", "value": format_rate(float_value(row, "hitter_hipa_75_ab"))},
            ],
            "matchup": [
                {"label": "H2H PA", "value": format_number(float_value(row, "h2h_pa"), digits=0)},
                {"label": "H2H xBA", "value": format_rate(float_value(row, "h2h_xba"))},
                {"label": "H2H K", "value": format_percent_rate(float_value(row, "h2h_k_rate"))},
                {"label": "Pitcher split", "value": format_rate(float_value(row, "pitcher_lr_opp_ba"))},
            ],
            "pitching": [
                {"label": "SP H/IP 350", "value": format_rate(float_value(row, "pitcher_hpi_350"))},
                {"label": "SP H/IP season", "value": format_rate(float_value(row, "pitcher_hpi_season"))},
                {"label": "Stuff+", "value": format_number(float_value(row, "pitcher_stuff_plus"), digits=1)},
                {"label": "Relief BA", "value": format_rate(bullpen_opp_ba)},
                {"label": "Bullpen H/IP", "value": format_rate(float_value(row, "bullpen_hpi"))},
            ],
            "context": [
                {"label": "Sprint", "value": format_number(float_value(row, "sprint_speed"), digits=1)},
                {"label": "Slot", "value": format_number(float_value(row, "lineup_slot"), digits=1)},
            ],
        },
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def available_dates(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row.get("date") for row in rows if row.get("date")})


def row_identity(row: dict[str, str]) -> tuple:
    game_pk = str(row.get("game_pk") or "").strip()
    player_id = str(row.get("player_id") or "").strip()
    if game_pk and player_id:
        return "game", game_pk, player_id
    return (
        "fallback",
        str(row.get("date") or "").strip(),
        normalize_name(row.get("player", "")),
        str(row.get("team") or "").strip(),
        str(row.get("opponent") or "").strip(),
    )


def select_dashboard_rows(
    source_rows: list[dict[str, str]],
    *,
    limit: int,
    congregation: dict[str, dict] | None,
) -> list[tuple[int, dict[str, str]]]:
    output = []
    seen = set()
    for model_rank, row in enumerate(source_rows, start=1):
        include = model_rank <= limit or congregation_record_for(row, congregation) is not None
        if not include:
            continue
        identity = row_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        output.append((model_rank, row))
    return output


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def update_dashboard_index(
    *,
    active_payload: dict,
    archive_dir: Path,
    index_json: Path,
) -> None:
    dashboards = []
    for path in sorted(archive_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dashboard_date = payload.get("date") or path.stem
        dashboards.append(
            {
                "date": dashboard_date,
                "path": f"dashboards/{path.name}",
                "generated_at": payload.get("generated_at") or "",
                "total_candidates": payload.get("total_candidates", 0),
                "pickable_count": payload.get("pickable_count", 0),
                "showing": payload.get("showing") or "",
            }
        )
    dashboards.sort(key=lambda row: row["date"], reverse=True)
    write_json(
        index_json,
        {
            "active_date": active_payload.get("date"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dashboards": dashboards,
        },
    )


def export_web_payload(
    *,
    candidates_csv: Path = DEFAULT_OUTPUT_CSV,
    out_json: Path = DEFAULT_WEB_JSON,
    index_json: Path = DEFAULT_DASHBOARD_INDEX,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    congregation_csv: Path = DEFAULT_CONGREGATION_CSV,
    target_date: str | None = None,
    limit: int = 10,
    fallback_latest: bool = True,
    archive: bool = True,
) -> dict:
    rows = load_rows(candidates_csv)
    dates = available_dates(rows)
    requested_date = target_date or date.today().isoformat()
    selected = [row for row in rows if row.get("date") == requested_date]
    selected_date = requested_date
    if not selected and fallback_latest and dates:
        selected_date = dates[-1]
        selected = [row for row in rows if row.get("date") == selected_date]

    pickable = [row for row in selected if str(row.get("pickable") or "").upper() == "Y"]
    source_rows = selected
    source_rows.sort(key=lambda row: float_value(row, "score") or 0.0, reverse=True)
    congregation = load_congregation(congregation_csv)
    display_rows = select_dashboard_rows(source_rows, limit=limit, congregation=congregation)
    congregation_shown = sum(
        1 for _, row in display_rows
        if congregation_record_for(row, congregation) is not None
    )
    selected_day = date.fromisoformat(selected_date) if selected_date else None
    game_states = build_game_state_lookup(selected_day, [row for _, row in display_rows])
    bullpen_stats = build_bullpen_stat_lookup(selected_day, [row for _, row in display_rows])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_date": requested_date,
        "date": selected_date,
        "used_latest_fallback": selected_date != requested_date,
        "total_candidates": len(selected),
        "pickable_count": len(pickable),
        "showing": "top_scored_candidates",
        "limit": limit,
        "congregation_count": congregation_shown,
        "picks": [
            candidate_payload(row, model_rank, game_states, bullpen_stats, congregation)
            for model_rank, row in display_rows
        ],
    }
    write_json(out_json, payload)
    if archive and payload.get("date"):
        archive_path = archive_dir / f"{payload['date']}.json"
        write_json(archive_path, payload)
        update_dashboard_index(active_payload=payload, archive_dir=archive_dir, index_json=index_json)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Export Statbirt top picks JSON for the web dashboard.")
    parser.add_argument("--candidates-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--out-json", default=str(DEFAULT_WEB_JSON))
    parser.add_argument("--index-json", default=str(DEFAULT_DASHBOARD_INDEX))
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--congregation-csv", default=str(DEFAULT_CONGREGATION_CSV))
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--all-dates", action="store_true", help="Export one dashboard JSON for every date in the candidates CSV.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--no-fallback-latest", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates_csv = Path(args.candidates_csv)
    target_dates = [args.date]
    if args.all_dates:
        target_dates = available_dates(load_rows(candidates_csv))
    payloads = []
    for target_date in target_dates:
        payloads.append(
            export_web_payload(
                candidates_csv=candidates_csv,
                out_json=Path(args.out_json),
                index_json=Path(args.index_json),
                archive_dir=Path(args.archive_dir),
                congregation_csv=Path(args.congregation_csv),
                target_date=target_date,
                limit=args.limit,
                fallback_latest=not args.no_fallback_latest,
                archive=not args.no_archive,
            )
        )
    latest = payloads[-1] if payloads else {"picks": [], "date": ""}
    print(
        f"Exported {len(payloads)} dashboard file(s); active board has "
        f"{len(latest['picks'])} web pick(s) for {latest['date']} at {Path(args.out_json).resolve()}."
    )


if __name__ == "__main__":
    main()
