from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

from .config import DEFAULT_STUFF_PLUS_CSV
from .fangraphs import _name_from_row, _stuff_value_from_row


def _team_from_row(row: dict[str, str]) -> str:
    for key in ("Team", "team", "Tm", "team_name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _player_id_from_row(row: dict[str, str]) -> str:
    for key in ("player_id", "mlbam_id", "MLBAMID", "MLBAM ID", "MLBAM"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def import_stuff_plus_csv(input_csv: str | Path, output_csv: str | Path, *, season: int) -> int:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    rows = []
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = _name_from_row(row)
            stuff_plus = _stuff_value_from_row(row)
            if not player or stuff_plus is None:
                continue
            rows.append(
                {
                    "season": str(season),
                    "player_id": _player_id_from_row(row),
                    "player": player,
                    "team": _team_from_row(row),
                    "stuff_plus": f"{stuff_plus:.1f}",
                    "source": str(input_csv),
                    "updated": date.today().isoformat(),
                }
            )

    rows.sort(key=lambda row: row["player"].lower())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["season", "player_id", "player", "team", "stuff_plus", "source", "updated"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize a FanGraphs Stuff+ leaderboard export for Statbirt.")
    parser.add_argument("input_csv", help="Downloaded FanGraphs leaderboard CSV.")
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--out", default=str(DEFAULT_STUFF_PLUS_CSV))
    return parser.parse_args()


def main():
    args = parse_args()
    count = import_stuff_plus_csv(args.input_csv, args.out, season=args.season)
    print(f"Wrote {count} Stuff+ rows to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

