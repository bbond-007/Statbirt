from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from .config import DEFAULT_OUTPUT_CSV
from .mlb_api import MLBClient, compute_bullpen_stats, season_start_for
from .results import load_candidates_table, parse_results_date, write_candidates_table
from .utils import format_float, parse_int


def ensure_bullpen_fields(fieldnames: list[str]) -> list[str]:
    output = list(fieldnames)
    if "bullpen_hpi" not in output:
        output.append("bullpen_hpi")
    if "bullpen_opp_ba" not in output:
        try:
            insert_at = output.index("bullpen_hpi") + 1
        except ValueError:
            insert_at = len(output)
        output.insert(insert_at, "bullpen_opp_ba")
    return output


def update_bullpen_csv(
    path: str | Path,
    *,
    refresh_filled: bool = False,
    dry_run: bool = False,
) -> dict:
    rows, fieldnames = load_candidates_table(path)
    fieldnames = ensure_bullpen_fields(fieldnames)
    pending_rows = [
        row for row in rows
        if parse_results_date(row.get("date"))
        and parse_int(row.get("opponent_id")) is not None
        and (refresh_filled or not str(row.get("bullpen_opp_ba", "")).strip())
    ]

    client = MLBClient()
    stats_by_date: dict = {}
    warnings: list[str] = []
    for row_date in sorted({parse_results_date(row.get("date")) for row in pending_rows if parse_results_date(row.get("date"))}):
        season_start = season_start_for(client, row_date.year)
        end_date = row_date - timedelta(days=1)
        if end_date < season_start:
            stats_by_date[row_date] = {}
            continue
        try:
            stats_by_date[row_date] = compute_bullpen_stats(client, season_start, end_date, row_date.year)
        except Exception as exc:
            warnings.append(f"{row_date.isoformat()}: bullpen stats failed: {type(exc).__name__}: {exc}")
            stats_by_date[row_date] = {}

    updated = 0
    missing = 0
    for row in pending_rows:
        row_date = parse_results_date(row.get("date"))
        opponent_id = parse_int(row.get("opponent_id"))
        values = (stats_by_date.get(row_date) or {}).get(opponent_id) if row_date and opponent_id else None
        if not values:
            missing += 1
            continue
        opp_ba = values.get("opponent_batting_average")
        if opp_ba is None:
            missing += 1
            continue
        updated += 1
        if not dry_run:
            row["bullpen_opp_ba"] = format_float(opp_ba, 3)
            if refresh_filled or not str(row.get("bullpen_hpi", "")).strip():
                row["bullpen_hpi"] = format_float(values.get("hits_per_inning"), 3)

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
    parser = argparse.ArgumentParser(description="Update Statbirt candidate bullpen relief pitching columns.")
    parser.add_argument("--candidates-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--refresh-filled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = update_bullpen_csv(
        Path(args.candidates_csv),
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
