from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

from .config import DEFAULT_OUTPUT_CSV, DEFAULT_STUFF_PLUS_CSV, PipelineConfig, StopValveConfig
from .learned_model import score_candidates, train_model
from .mlb_api import MLBClient, canonical_date, season_start_for
from .pipeline import build_daily_candidates, upsert_candidates_csv
from .results import update_results_csv
from .utils import normalize_name, parse_int


def existing_candidate_dates(path: str | Path) -> set[date]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    output = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            parsed = canonical_date(row.get("date"))
            if parsed is not None:
                output.add(parsed)
    return output


def regular_season_dates(client: MLBClient, start_day: date, end_day: date) -> list[date]:
    if end_day < start_day:
        return []
    games = client.schedule(start_day, end_day, hydrate="")
    dates = {
        parsed
        for game in games
        if game.get("gameType") == "R"
        for parsed in [canonical_date(game.get("officialDate") or game.get("gameDate"))]
        if parsed is not None
    }
    return sorted(dates)


def historical_lineups_for_date(client: MLBClient, target_day: date) -> dict[int, dict]:
    output: dict[int, dict] = {}
    games = [game for game in client.schedule(target_day, target_day, hydrate="") if game.get("gameType") == "R"]
    for game in games:
        game_pk = parse_int(game.get("gamePk"))
        if game_pk is None:
            continue
        try:
            boxscore = client.boxscore(game_pk)
        except Exception:
            continue
        for side in ("away", "home"):
            team_box = ((boxscore.get("teams") or {}).get(side) or {})
            team_id = parse_int(((team_box.get("team") or {}).get("id")))
            if team_id is None:
                continue
            batting_order = [parse_int(player_id) for player_id in (team_box.get("battingOrder") or [])]
            batting_order = [player_id for player_id in batting_order if player_id is not None]
            players = team_box.get("players") or {}
            players_by_id = {player_id: slot for slot, player_id in enumerate(batting_order[:9], start=1)}
            players_by_name = {}
            for player_id, slot in players_by_id.items():
                player = players.get(f"ID{player_id}") or {}
                name = ((player.get("person") or {}).get("fullName") or "").strip()
                if name:
                    players_by_name[normalize_name(name)] = slot
            if players_by_id:
                output[team_id] = {
                    "confirmed": True,
                    "players_by_id": players_by_id,
                    "players_by_name": players_by_name,
                }
    return output


def select_backfill_dates(
    *,
    season: int,
    start_day: date | None,
    end_day: date | None,
    output_csv: str | Path,
    rerun_existing: bool,
    max_days: int | None,
) -> list[date]:
    client = MLBClient()
    season_start = season_start_for(client, season)
    today = date.today()
    start = start_day or season_start
    end = end_day or min(today - timedelta(days=1), date(season, 12, 31))
    candidates = regular_season_dates(client, start, end)
    if not rerun_existing:
        existing = existing_candidate_dates(output_csv)
        candidates = [day for day in candidates if day not in existing]
    if max_days is not None:
        candidates = candidates[: max(0, max_days)]
    return candidates


def backfill_dates(
    dates: list[date],
    *,
    output_csv: str | Path,
    config: PipelineConfig,
    skip_savant: bool,
    skip_weather: bool,
    update_results: bool,
    train_learned_model: bool,
    use_historical_lineups: bool,
) -> dict[str, int]:
    summary = {
        "dates": len(dates),
        "built": 0,
        "empty": 0,
        "warnings": 0,
        "inserted": 0,
        "updated": 0,
    }
    for index, target_day in enumerate(dates, start=1):
        print(f"\n[{index}/{len(dates)}] Backfilling {target_day.isoformat()}...", flush=True)
        confirmed_lineups_override = None
        if use_historical_lineups:
            confirmed_lineups_override = historical_lineups_for_date(MLBClient(), target_day)
            if confirmed_lineups_override:
                print(f"Loaded historical lineups for {len(confirmed_lineups_override)} teams.", flush=True)
        candidates, warnings = build_daily_candidates(
            target_day,
            config=config,
            skip_savant=skip_savant,
            skip_weather=skip_weather,
            confirmed_lineups_override=confirmed_lineups_override,
            verbose=True,
        )
        summary["warnings"] += len(warnings)
        if warnings:
            print("Warnings:", flush=True)
            for warning in warnings[:8]:
                print(f"- {warning}", flush=True)
            if len(warnings) > 8:
                print(f"- ... {len(warnings) - 8} more", flush=True)
        if not candidates:
            print("No candidates produced.", flush=True)
            summary["empty"] += 1
            continue
        write_summary = upsert_candidates_csv(output_csv, candidates)
        summary["built"] += 1
        summary["inserted"] += write_summary["inserted"]
        summary["updated"] += write_summary["updated"]
        print(
            f"Upserted {len(candidates)} candidates "
            f"({write_summary['inserted']} inserted, {write_summary['updated']} updated).",
            flush=True,
        )

    if update_results and dates:
        print("\nUpdating postgame results for candidate rows...", flush=True)
        result_summary = update_results_csv(output_csv, only_dates=set(dates))
        print(
            f"Results updater touched {result_summary['updated']} rows "
            f"({result_summary['pending']} pending).",
            flush=True,
        )

    if train_learned_model and dates:
        print("\nRetraining learned hit-probability model...", flush=True)
        train_model(output_csv)
        records = score_candidates(output_csv, date_filter="latest")
        print(f"Scored {len(records)} latest-date rows with the learned model.", flush=True)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill Statbirt candidate rows for historical MLB dates.")
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--start-date", help="First date to consider, YYYY-MM-DD. Defaults to MLB season start.")
    parser.add_argument("--end-date", help="Last date to consider, YYYY-MM-DD. Defaults to yesterday.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--max-days", type=int, help="Limit how many missing dates to run in this batch.")
    parser.add_argument("--rerun-existing", action="store_true", help="Rebuild dates already present in the CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the dates that would be backfilled.")
    parser.add_argument("--update-results", action="store_true", help="Run postgame result updater after backfill.")
    parser.add_argument("--train-learned-model", action="store_true", help="Retrain and score the learned model after backfill.")
    parser.add_argument(
        "--no-historical-lineups",
        action="store_true",
        help="Do not seed historical candidates from final boxscore batting orders.",
    )
    parser.add_argument("--min-starts-last-5", type=int, default=3)
    parser.add_argument("--savant-years", type=int, default=3)
    parser.add_argument("--hitter-play-log-seasons", type=int, default=7)
    parser.add_argument("--pitcher-game-log-seasons", type=int, default=6)
    parser.add_argument("--stuff-plus-csv", default=str(DEFAULT_STUFF_PLUS_CSV))
    parser.add_argument("--skip-savant", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-bullpen", action="store_true")
    parser.add_argument(
        "--use-fangraphs-fetch",
        action="store_true",
        help="Try the live FanGraphs API in addition to the manual Stuff+ CSV. Off by default for backfills.",
    )
    parser.add_argument("--strict-missing-stop-data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start_day = date.fromisoformat(args.start_date) if args.start_date else None
    end_day = date.fromisoformat(args.end_date) if args.end_date else None
    dates = select_backfill_dates(
        season=args.season,
        start_day=start_day,
        end_day=end_day,
        output_csv=args.out,
        rerun_existing=args.rerun_existing,
        max_days=args.max_days,
    )

    print(f"Selected {len(dates)} date(s) for backfill.")
    if dates:
        print(f"Range: {dates[0].isoformat()} to {dates[-1].isoformat()}")
    if args.dry_run:
        for target_day in dates:
            print(target_day.isoformat())
        return

    stop_valves = StopValveConfig(strict_missing_stop_data=args.strict_missing_stop_data)
    config = PipelineConfig(
        stop_valves=stop_valves,
        min_starts_last_5=args.min_starts_last_5,
        savant_years=args.savant_years,
        hitter_play_log_seasons_back=args.hitter_play_log_seasons,
        pitcher_game_log_seasons_back=args.pitcher_game_log_seasons,
        compute_bullpen=not args.skip_bullpen,
        use_weather=not args.skip_weather,
        use_fangraphs_fetch=args.use_fangraphs_fetch,
        stuff_plus_csv=Path(args.stuff_plus_csv),
        output_csv=Path(args.out),
    )
    summary = backfill_dates(
        dates,
        output_csv=args.out,
        config=config,
        skip_savant=args.skip_savant,
        skip_weather=args.skip_weather,
        update_results=args.update_results,
        train_learned_model=args.train_learned_model,
        use_historical_lineups=not args.no_historical_lineups,
    )
    print("\nBackfill summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
