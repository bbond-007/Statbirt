from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_OUTPUT_CSV
from .results import update_results_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Update Statbirt candidate result columns from MLB boxscores.")
    parser.add_argument("--results-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--refresh-filled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = update_results_csv(
        Path(args.results_csv),
        refresh_filled=args.refresh_filled,
        dry_run=args.dry_run,
    )
    action = "Would update" if args.dry_run else "Updated"
    print(
        f"{action} {summary['updated']} row(s) in {Path(args.results_csv).resolve()} "
        f"({summary['pending']} pending, {summary['rows']} total)."
    )


if __name__ == "__main__":
    main()

