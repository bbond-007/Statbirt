from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_OUTPUT_CSV
from .mlb_api import MLBClient, get_games_for_date
from .results import load_candidates_table, parse_results_date, write_candidates_table
from .utils import format_float, parse_int
from .weather import fetch_weather_forecast


def ensure_weather_fields(fieldnames: list[str]) -> list[str]:
    output = list(fieldnames)
    if "precip_probability" not in output:
        try:
            insert_at = output.index("doubleheader") + 1
        except ValueError:
            insert_at = len(output)
        output.insert(insert_at, "precip_probability")
    if "forecast_temperature_f" not in output:
        try:
            insert_at = output.index("precip_probability") + 1
        except ValueError:
            insert_at = len(output)
        output.insert(insert_at, "forecast_temperature_f")
    return output


def update_weather_csv(
    path: str | Path,
    *,
    date_filter=None,
    refresh_filled: bool = False,
    dry_run: bool = False,
) -> dict:
    rows, fieldnames = load_candidates_table(path)
    fieldnames = ensure_weather_fields(fieldnames)
    pending_rows = [
        row for row in rows
        if parse_results_date(row.get("date"))
        and (date_filter is None or parse_results_date(row.get("date")) == date_filter)
        and parse_int(row.get("game_pk")) is not None
        and (
            refresh_filled
            or not str(row.get("precip_probability", "")).strip()
            or not str(row.get("forecast_temperature_f", "")).strip()
        )
    ]
    dates = sorted({parse_results_date(row.get("date")) for row in pending_rows if parse_results_date(row.get("date"))})
    client = MLBClient()
    weather_by_game: dict[int, object] = {}
    warnings: list[str] = []

    for day in dates:
        try:
            games = get_games_for_date(client, day)
        except Exception as exc:
            warnings.append(f"{day.isoformat()}: game schedule lookup failed: {type(exc).__name__}: {exc}")
            continue
        for game in games:
            if game.game_pk is None:
                continue
            forecast, warning = fetch_weather_forecast(
                latitude=game.venue_latitude,
                longitude=game.venue_longitude,
                game_datetime_utc=game.game_datetime_utc,
            )
            weather_by_game[game.game_pk] = forecast
            if warning:
                warnings.append(f"{day.isoformat()} {game.away_abbr}@{game.home_abbr}: {warning}")

    updated = 0
    missing = 0
    for row in pending_rows:
        game_pk = parse_int(row.get("game_pk"))
        forecast = weather_by_game.get(game_pk) if game_pk is not None else None
        probability = getattr(forecast, "precipitation_probability", None)
        temperature_f = getattr(forecast, "temperature_f", None)
        if probability is None and temperature_f is None:
            missing += 1
            continue
        updated += 1
        if not dry_run:
            if probability is not None:
                row["precip_probability"] = format_float(probability, 1)
            if temperature_f is not None:
                row["forecast_temperature_f"] = format_float(temperature_f, 1)

    if updated and not dry_run:
        write_candidates_table(path, rows, fieldnames)

    return {
        "updated": updated,
        "pending": len(pending_rows),
        "missing": missing,
        "rows": len(rows),
        "warnings": warnings,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Update Statbirt candidate precipitation probability from venue weather.")
    parser.add_argument("--candidates-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--date", help="Only update rows for this game date, YYYY-MM-DD.")
    parser.add_argument("--refresh-filled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = update_weather_csv(
        Path(args.candidates_csv),
        date_filter=parse_results_date(args.date) if args.date else None,
        refresh_filled=args.refresh_filled,
        dry_run=args.dry_run,
    )
    action = "Would update" if args.dry_run else "Updated"
    print(
        f"{action} {summary['updated']} row(s) in {Path(args.candidates_csv).resolve()} "
        f"({summary['pending']} pending, {summary['missing']} missing, {summary['rows']} total)."
    )
    if summary["warnings"]:
        print("Warnings:")
        for warning in summary["warnings"][:12]:
            print(f"- {warning}")
        if len(summary["warnings"]) > 12:
            print(f"- ... {len(summary['warnings']) - 12} more")


if __name__ == "__main__":
    main()
