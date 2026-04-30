from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .config import DEFAULT_OUTPUT_CSV, DEFAULT_STUFF_PLUS_CSV, PipelineConfig, StopValveConfig
from .pipeline import build_daily_candidates, upsert_candidates_csv
from .utils import format_float


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Statbirt daily MLB hit-pick model.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path.")
    parser.add_argument("--top", type=int, default=25, help="Rows to print in the terminal summary.")
    parser.add_argument("--min-starts-last-5", type=int, default=3)
    parser.add_argument("--savant-years", type=int, default=3)
    parser.add_argument("--hitter-play-log-seasons", type=int, default=7)
    parser.add_argument("--pitcher-game-log-seasons", type=int, default=6)
    parser.add_argument("--stuff-plus-csv", default=str(DEFAULT_STUFF_PLUS_CSV))
    parser.add_argument("--skip-savant", action="store_true", help="Skip Savant matchup/split features.")
    parser.add_argument("--skip-weather", action="store_true", help="Skip Open-Meteo weather lookup.")
    parser.add_argument("--skip-fangraphs-fetch", action="store_true", help="Only use manual Stuff+ CSV.")
    parser.add_argument("--skip-bullpen", action="store_true", help="Skip boxscore-built bullpen H/IP.")
    parser.add_argument(
        "--strict-missing-stop-data",
        action="store_true",
        help="Treat missing stop-valve data as a hard pass instead of a concern.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    stop_valves = StopValveConfig(strict_missing_stop_data=args.strict_missing_stop_data)
    config = PipelineConfig(
        stop_valves=stop_valves,
        min_starts_last_5=args.min_starts_last_5,
        savant_years=args.savant_years,
        hitter_play_log_seasons_back=args.hitter_play_log_seasons,
        pitcher_game_log_seasons_back=args.pitcher_game_log_seasons,
        compute_bullpen=not args.skip_bullpen,
        use_weather=not args.skip_weather,
        use_fangraphs_fetch=not args.skip_fangraphs_fetch,
        stuff_plus_csv=Path(args.stuff_plus_csv),
        output_csv=Path(args.out),
    )
    candidates, warnings = build_daily_candidates(
        target_date,
        config=config,
        skip_savant=args.skip_savant,
        skip_weather=args.skip_weather,
        verbose=True,
    )
    write_summary = None
    if candidates:
        write_summary = upsert_candidates_csv(args.out, candidates)

    print()
    if warnings:
        print("Warnings:")
        for warning in warnings[:12]:
            print(f"- {warning}")
        if len(warnings) > 12:
            print(f"- ... {len(warnings) - 12} more")
        print()

    if not candidates:
        print("No candidates produced.")
        return

    print(f"Top {min(args.top, len(candidates))} Statbirt candidates for {target_date.isoformat()}")
    print("=" * 132)
    print(
        f"{'Player':<24} {'Tm':<3} {'Opp':<3} {'Pick':<4} {'Score':>6} {'Slot':>4} "
        f"{'HiPA2500':>8} {'PA/G':>5} {'SP H/IP':>7} {'Stuff+':>7} {'H2H PA':>6} {'Concerns':<36}"
    )
    print("-" * 132)
    for candidate in candidates[: args.top]:
        f = candidate.features
        concerns = "; ".join(candidate.valve_result.hard_pass_reasons[:1] or candidate.valve_result.concerns[:1])
        print(
            f"{f.player_name:<24} {f.team:<3} {f.opponent:<3} "
            f"{'Y' if candidate.valve_result.pickable else 'N':<4} "
            f"{candidate.score:>6.2f} "
            f"{(format_float(f.lineup_slot, 1) or 'N/A'):>4} "
            f"{(format_float(f.hitter_hipa_2500_pa, 3) or 'N/A'):>8} "
            f"{(format_float(f.hitter_pa_per_game_season, 2) or 'N/A'):>5} "
            f"{(format_float(f.pitcher_hpi_season, 3) or 'N/A'):>7} "
            f"{(format_float(f.pitcher_stuff_plus, 1) or 'N/A'):>7} "
            f"{f.h2h_pa:>6} "
            f"{concerns:<36}"
        )
    if write_summary:
        print(
            f"\nUpserted {len(candidates)} rows into {Path(args.out).resolve()} "
            f"({write_summary['inserted']} inserted, {write_summary['updated']} updated, "
            f"{write_summary['total_rows']} total)."
        )


if __name__ == "__main__":
    main()
