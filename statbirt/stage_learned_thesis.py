from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

from .config import DATA_DIR, DEFAULT_OUTPUT_CSV, MANUAL_DIR
from .export_learned_web import (
    build_pick_payload,
    candidate_lookup,
    merge_prediction_with_candidate,
    rows_by_date,
    sort_predictions,
)
from .export_web import DEFAULT_CONGREGATION_CSV, load_congregation, load_rows, row_identity, split_reasons, write_json
from .injuries import current_injured_player_ids_for_rows, filter_injured_rows
from .learned_model import DEFAULT_PREDICTIONS_CSV
from .learned_selection import build_selection_brief

DEFAULT_CONTEXT_DIR = MANUAL_DIR / "learned_top2_context"
DEFAULT_THESIS_DIR = MANUAL_DIR / "learned_top2_theses"
DEFAULT_WORKFLOW_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "learned_top2_thesis.md"

FIELD_GROUPS = {
    "identity": [
        "date",
        "player",
        "player_id",
        "team",
        "team_id",
        "opponent",
        "opponent_id",
        "game_pk",
    ],
    "learned_model": [
        "learned_rank",
        "learned_hit_probability",
        "model_version",
        "model_trained_at",
        "bob_score",
        "score",
        "pickable",
    ],
    "game_context": [
        "venue_name",
        "game_start_time_utc",
        "road_game",
        "division_matchup",
        "doubleheader",
        "confirmed_lineup",
        "lineup_slot",
        "expected_pa",
        "starts_last_5",
    ],
    "weather_park": [
        "precip_probability",
        "forecast_temperature_f",
        "park_hit_factor",
    ],
    "hitter_form_contact": [
        "hitter_ba_season",
        "hitter_hipa_2500_pa",
        "hitter_pa_per_game_season",
        "hitter_ba_2500_ab",
        "hitter_hipa_500_pa",
        "hitter_hipa_75_ab",
        "hitter_ba_75_ab",
        "hitter_ba_25_ab",
        "hitter_ba_500_ab",
        "hitter_last_5_games_played",
        "hitter_last_5_games_hits",
        "hitter_last_5_games_ab",
        "hitter_last_5_games_ba",
        "sprint_speed",
    ],
    "hitter_plate_discipline": [
        "hitter_bb_rate_season",
        "hitter_bb_rate_500_pa",
        "hitter_whiff_rate_season",
        "hitter_whiff_rate_500_pa",
        "hitter_k_rate_season",
        "hitter_k_rate_500_pa",
    ],
    "hitter_handedness_splits": [
        "batter_stand",
        "pitcher_hand",
        "hitter_split_ba_season_vs_lhp",
        "hitter_split_ba_season_vs_rhp",
        "hitter_split_pa_season_vs_lhp",
        "hitter_split_pa_season_vs_rhp",
        "hitter_split_ba_500_vs_lhp",
        "hitter_split_ba_500_vs_rhp",
        "hitter_split_ba_1500_vs_lhp",
        "hitter_split_ba_1500_vs_rhp",
    ],
    "probable_starter": [
        "probable_pitcher",
        "probable_pitcher_id",
        "pitcher_hand",
        "pitcher_hpi_350",
        "pitcher_hpi_200",
        "pitcher_hpi_season",
        "pitcher_hits_last_18_ip",
        "pitcher_stuff_plus",
        "pitcher_lr_opp_ba",
        "pitcher_lr_opp_ba_50",
        "pitcher_lr_opp_ba_200",
    ],
    "probable_starter_last_start": [
        "pitcher_last_start_date",
        "pitcher_last_start_ip",
        "pitcher_last_start_hits",
        "pitcher_last_start_strikeouts",
        "pitcher_last_start_walks",
    ],
    "h2h": [
        "h2h_pa",
        "h2h_hits",
        "h2h_hit_rate",
        "h2h_whiff_rate",
        "h2h_k_rate",
        "h2h_exit_velocity",
        "h2h_xba",
    ],
    "pitch_type_matchup": [
        "inferred_pitch_type_ba",
        "inferred_pitch_type_xba",
        "inferred_pitch_type_coverage",
    ],
    "bullpen_context": [
        "bullpen_hpi",
        "bullpen_opp_ba",
    ],
    "bob_score_components": [
        "component_hitter.hipa_2500_pa",
        "component_hitter.pa_per_game_season",
        "component_hitter.hipa_500_pa",
        "component_hitter.hipa_75_ab",
        "component_starting_pitcher.hpi_350",
        "component_starting_pitcher.hpi_season",
        "component_starting_pitcher.stuff_plus",
        "component_h2h.direct",
        "component_h2h.pitcher_lr_opp_ba",
        "component_h2h.inferred_pitch_type",
        "component_bullpen.hpi_season",
        "component_other.road_game",
        "component_other.division_matchup",
        "component_other.sprint_speed",
        "component_other.park_hit_factor",
        "component_other.lineup_opportunity",
    ],
    "results": [
        "result_hit",
        "result_hits",
        "result_ab",
        "result_pa",
        "result_status",
        "result_updated_at",
        "notes",
    ],
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(value: str) -> str:
    return "-".join(part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split() if part)


def _compact_fields(row: dict[str, str], fields: list[str]) -> dict[str, str]:
    return {
        field: str(row.get(field) or "").strip()
        for field in fields
        if str(row.get(field) or "").strip()
    }


def _missing_fields(row: dict[str, str], fields: list[str]) -> list[str]:
    return [field for field in fields if not str(row.get(field) or "").strip()]


def _source_targets_for(row: dict[str, str], pick: dict) -> list[dict[str, str]]:
    player = pick.get("player") or row.get("player") or ""
    player_id = pick.get("player_id") or row.get("player_id") or ""
    pitcher = pick.get("probable_pitcher") or row.get("probable_pitcher") or ""
    pitcher_id = row.get("probable_pitcher_id") or ""
    game_pk = pick.get("game_pk") or row.get("game_pk") or ""
    team = pick.get("team") or row.get("team") or ""
    opponent = pick.get("opponent") or row.get("opponent") or ""

    targets = [
        {
            "label": "MLB probable pitchers",
            "url": "https://www.mlb.com/probable-pitchers",
            "why": "Verify the probable starter has not changed.",
        },
        {
            "label": "MLB starting lineups",
            "url": "https://www.mlb.com/starting-lineups",
            "why": "Check whether the hitter is confirmed in the lineup and whether the slot changed.",
        },
        {
            "label": "MLB injury report",
            "url": "https://www.mlb.com/injury-report",
            "why": "Check for late injury or availability notes.",
        },
        {
            "label": "RotoWire bullpen usage",
            "url": "https://www.rotowire.com/baseball/bullpen-usage.php",
            "why": "Look for bullpen fatigue or unavailable leverage arms.",
        },
        {
            "label": "RotoWire daily lineups",
            "url": "https://www.rotowire.com/baseball/daily-lineups.php",
            "why": "Cross-check lineup and late-scratch risk.",
        },
        {
            "label": "MLB news",
            "url": "https://www.mlb.com/news",
            "why": f"Search for current notes on {player}, {pitcher}, {team}, and {opponent}.",
        },
    ]
    if game_pk:
        targets.extend(
            [
                {
                    "label": "MLB live game feed",
                    "url": f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
                    "why": "Verify official game, lineup, venue, and probable-pitcher context.",
                },
                {
                    "label": "MLB boxscore endpoint",
                    "url": f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore",
                    "why": "Useful after lineups post or for later result checks.",
                },
            ]
        )
    if player and player_id:
        targets.append(
            {
                "label": f"Baseball Savant: {player}",
                "url": f"https://baseballsavant.mlb.com/savant-player/{_slug(player)}-{player_id}",
                "why": "Review hitter quality of contact, rolling form, pitch-type strengths, and handedness context.",
            }
        )
    if pitcher and pitcher_id:
        targets.append(
            {
                "label": f"Baseball Savant: {pitcher}",
                "url": f"https://baseballsavant.mlb.com/savant-player/{_slug(pitcher)}-{pitcher_id}",
                "why": "Review starter pitch mix, recent shape, contact allowed, and possible matchup fragility.",
            }
        )
    return targets


def _player_context(prediction: dict[str, str], merged: dict[str, str], pick: dict, rank: int) -> dict:
    field_context = {
        group: _compact_fields(merged, fields)
        for group, fields in FIELD_GROUPS.items()
    }
    missing_relevant_fields = {
        group: missing
        for group, fields in FIELD_GROUPS.items()
        if (missing := _missing_fields(merged, fields))
    }
    stop_valves = pick.get("hard_pass_reasons") or split_reasons(merged.get("hard_pass_reasons"))
    watch_notes = pick.get("concerns") or split_reasons(merged.get("concerns"))
    return {
        "rank": pick.get("learned_rank") or rank,
        "player": pick.get("player") or merged.get("player") or prediction.get("player") or "",
        "player_id": pick.get("player_id") or merged.get("player_id") or prediction.get("player_id") or "",
        "team": pick.get("team") or merged.get("team") or "",
        "opponent": pick.get("opponent") or merged.get("opponent") or "",
        "dashboard_summary": {
            "learned_hit_probability": pick.get("learned_hit_probability"),
            "bob_score": pick.get("bob_score", pick.get("score")),
            "pickable": pick.get("pickable"),
            "safety_score": pick.get("safety_score"),
            "h2h_record": pick.get("h2h_record"),
            "expected_pa": pick.get("expected_pa"),
            "lineup_slot": pick.get("lineup_slot"),
            "season_ba": pick.get("hitter_ba_season"),
            "last_5_ba": pick.get("hitter_last_5_games_ba"),
            "hot_streak": pick.get("hot_streak"),
            "hot_streak_tooltip": pick.get("hot_streak_tooltip"),
            "probable_pitcher": pick.get("probable_pitcher"),
            "pitcher_hand": pick.get("pitcher_hand"),
            "batter_stand": pick.get("batter_stand"),
            "venue_name": pick.get("venue_name"),
            "weather_label": pick.get("weather_label"),
            "congregation_status": pick.get("congregation_status"),
        },
        "model_signals": pick.get("selection_signals") or [],
        "model_risks": pick.get("selection_risks") or [],
        "stop_valves": stop_valves,
        "watch_notes": watch_notes,
        "field_context": field_context,
        "missing_relevant_fields": missing_relevant_fields,
        "source_targets": _source_targets_for(merged, pick),
        "chatgpt_review_questions": [
            "Has the probable starter or lineup changed since the morning run?",
            "Do current news, beat notes, or injury reports create scratch, rest, or pinch-hit risk?",
            "Does the hitter have a pitch-type or handedness edge that survives beyond the headline model probability?",
            "Is the opposing bullpen likely to help or hurt the one-hit path if the starter exits early?",
            "Is the recent form signal supported by contact quality, or does it look like short-sample noise?",
        ],
    }


def _selected_rows(
    *,
    predictions_by_date: dict[str, list[dict[str, str]]],
    candidates_by_date: dict[str, list[dict[str, str]]],
    target_date: str | None,
    top: int,
    fallback_latest: bool,
    filter_injured: bool,
    injured_player_ids: set[int] | frozenset[int] | None,
) -> tuple[str, str, list[dict[str, str]], list[dict[str, str]], int]:
    dates = sorted(predictions_by_date)
    if target_date == "latest":
        requested_date = dates[-1] if dates else date.today().isoformat()
    else:
        requested_date = target_date or date.today().isoformat()
    selected_date = requested_date
    selected_predictions = list(predictions_by_date.get(selected_date, []))
    if not selected_predictions and fallback_latest and dates:
        selected_date = dates[-1]
        selected_predictions = list(predictions_by_date.get(selected_date, []))

    if filter_injured:
        selected_predictions, injury_filtered_count = filter_injured_rows(selected_predictions, injured_player_ids or set())
    else:
        injury_filtered_count = 0

    selected_predictions = sort_predictions(selected_predictions)[:top]
    selected_candidates = list(candidates_by_date.get(selected_date, []))
    if filter_injured:
        selected_candidates, _ = filter_injured_rows(selected_candidates, injured_player_ids or set())
    return requested_date, selected_date, selected_predictions, selected_candidates, injury_filtered_count


def build_thesis_context(
    *,
    predictions_csv: Path = DEFAULT_PREDICTIONS_CSV,
    candidates_csv: Path = DEFAULT_OUTPUT_CSV,
    congregation_csv: Path = DEFAULT_CONGREGATION_CSV,
    target_date: str | None = "latest",
    top: int = 2,
    fallback_latest: bool = True,
    filter_injured: bool = True,
    injured_player_ids: set[int] | frozenset[int] | None = None,
    workflow_prompt: Path = DEFAULT_WORKFLOW_PROMPT,
    thesis_dir: Path = DEFAULT_THESIS_DIR,
) -> dict:
    if top < 1:
        raise ValueError("--top must be at least 1.")
    prediction_rows = load_rows(predictions_csv)
    candidate_rows = load_rows(candidates_csv)
    if filter_injured and injured_player_ids is None:
        injured_player_ids = current_injured_player_ids_for_rows(candidate_rows)

    predictions_by_date = rows_by_date(prediction_rows)
    candidates_by_date = rows_by_date(candidate_rows)
    requested_date, selected_date, selected_predictions, selected_candidates, injury_filtered_count = _selected_rows(
        predictions_by_date=predictions_by_date,
        candidates_by_date=candidates_by_date,
        target_date=target_date,
        top=top,
        fallback_latest=fallback_latest,
        filter_injured=filter_injured,
        injured_player_ids=injured_player_ids,
    )
    lookup = candidate_lookup(selected_candidates)
    congregation = load_congregation(congregation_csv)

    picks = [
        build_pick_payload(prediction, lookup, rank, congregation)
        for rank, prediction in enumerate(selected_predictions, start=1)
    ]
    players = []
    for rank, (prediction, pick) in enumerate(zip(selected_predictions, picks, strict=True), start=1):
        candidate = lookup.get(row_identity(prediction), {})
        merged = merge_prediction_with_candidate(prediction, candidate)
        players.append(_player_context(prediction, merged, pick, rank))

    return {
        "generated_at": _now_utc(),
        "date": selected_date,
        "requested_date": requested_date,
        "used_latest_fallback": selected_date != requested_date,
        "target": f"learned_rank_top_{top}",
        "purpose": "Evidence staging only. ChatGPT/Codex should write the pro thesis, con thesis, and committee decision.",
        "daily_timing": {
            "morning_models_scheduled": "07:00 America/Chicago",
            "recommended_stage_after": "08:00 America/Chicago",
            "note": "The morning model scripts normally need about 45 minutes, so 8:00am Central gives them a buffer.",
        },
        "workflow_prompt": str(workflow_prompt),
        "expected_thesis_path": str((thesis_dir / f"{selected_date}.json").resolve()) if selected_date else "",
        "dashboard_refresh_command": "py -3 -m statbirt.export_learned_web --date latest --limit 5",
        "total_predictions_for_date": len(predictions_by_date.get(selected_date, [])),
        "injury_filtered_count": injury_filtered_count,
        "selection_brief_from_local_metrics": build_selection_brief(picks),
        "players": players,
        "chatgpt_committee_process": [
            "Read this staged context and treat it as evidence, not as a thesis.",
            "Browse current free sources for lineup, probable pitcher, injury/news, bullpen, and matchup updates.",
            "Write a pro thesis and con thesis for each player.",
            "Have a final reviewer compare the arguments and choose one single hitter.",
            "Save the finished dashboard-ready JSON to expected_thesis_path, then refresh the learned dashboard export.",
        ],
        "output_schema": {
            "date": selected_date,
            "target": f"learned_rank_top_{top}",
            "committee_pick": "Player Name",
            "committee_summary": "One paragraph explaining the single-pick decision.",
            "source_notes": [{"label": "Source name", "url": "https://...", "note": "What was reviewed."}],
            "players": [
                {
                    "rank": 1,
                    "player": "Player Name",
                    "team": "AAA",
                    "opponent": "BBB",
                    "probable_pitcher": "Pitcher Name",
                    "pro_thesis": "Dashboard-ready paragraph.",
                    "con_thesis": "Dashboard-ready paragraph.",
                    "committee_thesis": "Dashboard-ready paragraph.",
                }
            ],
        },
    }


def stage_thesis_context(
    *,
    out_dir: Path = DEFAULT_CONTEXT_DIR,
    **kwargs,
) -> tuple[dict, Path]:
    payload = build_thesis_context(**kwargs)
    output_path = out_dir / f"{payload['date']}.json"
    write_json(output_path, payload)
    return payload, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage factual context for a ChatGPT-written learned top-2 thesis.")
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV))
    parser.add_argument("--candidates-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--congregation-csv", default=str(DEFAULT_CONGREGATION_CSV))
    parser.add_argument("--date", default="latest", help="YYYY-MM-DD or latest.")
    parser.add_argument("--top", type=int, default=2)
    parser.add_argument("--out-dir", default=str(DEFAULT_CONTEXT_DIR))
    parser.add_argument("--thesis-dir", default=str(DEFAULT_THESIS_DIR))
    parser.add_argument("--workflow-prompt", default=str(DEFAULT_WORKFLOW_PROMPT))
    parser.add_argument("--no-fallback-latest", action="store_true")
    parser.add_argument("--include-injured", action="store_true", help="Do not filter players currently listed as injured.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, output_path = stage_thesis_context(
        predictions_csv=Path(args.predictions_csv),
        candidates_csv=Path(args.candidates_csv),
        congregation_csv=Path(args.congregation_csv),
        target_date=args.date,
        top=args.top,
        fallback_latest=not args.no_fallback_latest,
        filter_injured=not args.include_injured,
        workflow_prompt=Path(args.workflow_prompt),
        thesis_dir=Path(args.thesis_dir),
        out_dir=Path(args.out_dir),
    )
    players = ", ".join(player["player"] for player in payload.get("players", []))
    print(
        f"Staged {payload['target']} thesis context for {payload['date']} "
        f"({players}) at {output_path.resolve()}."
    )


if __name__ == "__main__":
    main()
