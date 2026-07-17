from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .config import DATA_DIR, DEFAULT_OUTPUT_CSV
from .learned_model import DEFAULT_PREDICTIONS_CSV
from .utils import parse_float, parse_int

DEFAULT_SELECTION_BACKTEST_JSON = DATA_DIR / "models" / "learned_selection_backtest.json"
POLICY_VERSION = "learned-selection-v1"


def _number(value, default: float = 0.0) -> float:
    parsed = parse_float(value)
    return default if parsed is None else parsed


def _probability(pick: dict) -> float:
    return _number(pick.get("learned_hit_probability"))


def _rate(value) -> float | None:
    parsed = parse_float(value)
    return parsed


def _int(value, default: int = 0) -> int:
    parsed = parse_int(value)
    return default if parsed is None else parsed


def _is_yes(value) -> bool:
    return str(value or "").strip().lower() in {"y", "yes", "true", "1"}


def _rank(pick: dict) -> int:
    return _int(pick.get("learned_rank") or pick.get("rank"), 999999)


def _player_label(pick: dict) -> str:
    return str(pick.get("player") or "Unknown player")


def h2h_rate(pick: dict) -> float | None:
    pa = _int(pick.get("h2h_pa"))
    if pa <= 0:
        return None
    return _number(pick.get("h2h_hits")) / pa


def selection_score(pick: dict) -> float:
    probability = _probability(pick) * 100.0
    safety = _number(pick.get("safety_score"), 50.0)
    bob_score = _number(pick.get("bob_score", pick.get("score")), 50.0)
    expected_pa = _number(pick.get("expected_pa"), 4.0)
    lineup_slot = _number(pick.get("lineup_slot"), 6.0)
    h2h_pa = _int(pick.get("h2h_pa"))
    h2h = h2h_rate(pick)
    rank_penalty = max(0, _rank(pick) - 1) * 0.65
    risk_penalty = min(8, len(pick.get("selection_risks") or []) * 1.4)

    opportunity_bonus = 0.0
    if expected_pa >= 4.5:
        opportunity_bonus += 3.5
    elif expected_pa >= 4.2:
        opportunity_bonus += 1.5
    elif expected_pa < 4.0:
        opportunity_bonus -= 3.0
    if lineup_slot <= 3:
        opportunity_bonus += 2.0
    elif lineup_slot >= 7:
        opportunity_bonus -= 2.0
    if _is_yes(pick.get("confirmed_lineup")):
        opportunity_bonus += 1.0

    contact_bonus = 0.0
    season_ba = _rate(pick.get("hitter_ba_season"))
    recent_ba = _rate(pick.get("hitter_last_5_games_ba"))
    if season_ba is not None and season_ba >= 0.300:
        contact_bonus += 2.0
    elif season_ba is not None and season_ba < 0.240:
        contact_bonus -= 2.5
    if recent_ba is not None and recent_ba >= 0.400:
        contact_bonus += 1.5

    matchup_bonus = 0.0
    if h2h is not None and h2h_pa >= 3:
        if h2h >= 0.300:
            matchup_bonus += 2.0
        elif h2h <= 0.100:
            matchup_bonus -= 1.5
    elif h2h_pa == 0:
        matchup_bonus -= 0.8

    score = (
        probability * 0.54
        + safety * 0.24
        + bob_score * 0.14
        + opportunity_bonus
        + contact_bonus
        + matchup_bonus
        - risk_penalty
        - rank_penalty
    )
    return round(max(0.0, min(100.0, score)), 1)


def _format_rate(value) -> str:
    parsed = _rate(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.3f}".replace("0.", ".")


def _confidence_label(score: float) -> str:
    if score >= 78:
        return "Primary"
    if score >= 72:
        return "Strong"
    if score >= 66:
        return "Playable"
    return "Watch"


def _sentence_fragment(text: str) -> str:
    fragment = str(text or "").strip().rstrip(".")
    if not fragment:
        return ""
    return fragment[:1].lower() + fragment[1:]


def _pros_for(pick: dict) -> list[str]:
    pros: list[str] = []
    probability = _probability(pick)
    safety = _number(pick.get("safety_score"), 0)
    expected_pa = _number(pick.get("expected_pa"), 0)
    lineup_slot = _number(pick.get("lineup_slot"), 99)
    season_ba = _rate(pick.get("hitter_ba_season"))
    h2h = h2h_rate(pick)

    if probability >= 0.72:
        pros.append(f"High learned probability ({probability * 100:.1f}%).")
    elif probability >= 0.68:
        pros.append(f"Solid learned probability ({probability * 100:.1f}%).")
    if safety >= 75:
        pros.append(f"Strong reliability score ({safety:.0f}/100).")
    if expected_pa >= 4.4:
        pros.append(f"Projected for {expected_pa:.1f} PA.")
    if lineup_slot <= 3:
        pros.append(f"Premium lineup slot ({lineup_slot:.0f}).")
    if season_ba is not None and season_ba >= 0.285:
        pros.append(f"Current-season BA {_format_rate(season_ba)}.")
    if pick.get("hot_streak"):
        pros.append(f"Hot last-5 line ({pick.get('hot_streak_tooltip')}).")
    if h2h is not None and _int(pick.get("h2h_pa")) >= 3 and h2h >= 0.300:
        pros.append(f"Positive H2H record ({pick.get('h2h_record')}).")
    elif _int(pick.get("h2h_pa")) > 0:
        pros.append(f"Has seen the probable starter ({pick.get('h2h_record')}).")
    if pick.get("selection_signals"):
        for signal in pick["selection_signals"]:
            if signal not in " ".join(pros):
                pros.append(str(signal) + ".")
            if len(pros) >= 4:
                break
    return pros[:4] or ["Model keeps him in the learned top 5."]


def _cons_for(pick: dict) -> list[str]:
    cons: list[str] = []
    safety = _number(pick.get("safety_score"), 100)
    expected_pa = _number(pick.get("expected_pa"), 5)
    lineup_slot = _number(pick.get("lineup_slot"), 1)
    season_ba = _rate(pick.get("hitter_ba_season"))
    h2h = h2h_rate(pick)
    h2h_pa = _int(pick.get("h2h_pa"))

    if h2h_pa == 0:
        cons.append("No H2H history against the probable starter.")
    elif h2h is not None and h2h_pa >= 3 and h2h <= 0.100:
        cons.append(f"Thin H2H hit record ({pick.get('h2h_record')}).")
    if expected_pa < 4.1:
        cons.append(f"Lower PA projection ({expected_pa:.1f}).")
    if lineup_slot >= 7:
        cons.append(f"Lower lineup slot ({lineup_slot:.0f}).")
    if safety < 62:
        cons.append(f"Reliability score is only {safety:.0f}/100.")
    if season_ba is not None and season_ba < 0.240:
        cons.append(f"Current-season BA is {_format_rate(season_ba)}.")
    for risk in pick.get("selection_risks") or []:
        text = str(risk)
        if text and text not in " ".join(cons):
            cons.append(text + ".")
        if len(cons) >= 4:
            break
    if not cons and pick.get("hard_pass_reasons"):
        cons.append(str(pick["hard_pass_reasons"][0]) + ".")
    return cons[:4] or ["No major model risk flags inside the top 5 context."]


def _thesis_for(pick: dict, score: float, pros: list[str], cons: list[str]) -> str:
    player = _player_label(pick)
    probability = _probability(pick)
    safety = _number(pick.get("safety_score"), 0)
    expected_pa = _number(pick.get("expected_pa"), 0)
    primary_case = _sentence_fragment(pros[0] if pros else "the model keeps him in the top 5")
    primary_risk = _sentence_fragment(cons[0] if cons else "there are no major model risk flags")

    return (
        f"{player} is a {score:.1f} selection-score play with a "
        f"{probability * 100:.1f}% learned hit probability, {expected_pa:.1f} expected PA, "
        f"and a {safety:.0f}/100 reliability score. The positive case starts with {primary_case}; "
        f"the main counterweight is {primary_risk}."
    )


def build_pick_brief(pick: dict) -> dict:
    score = selection_score(pick)
    pros = _pros_for(pick)
    cons = _cons_for(pick)
    return {
        "rank": _rank(pick),
        "player": _player_label(pick),
        "team": pick.get("team") or "",
        "opponent": pick.get("opponent") or "",
        "selection_score": score,
        "learned_hit_probability": round(_probability(pick), 4),
        "pros": pros,
        "cons": cons,
        "confidence_label": _confidence_label(score),
        "thesis": _thesis_for(pick, score, pros, cons),
        "summary": f"{_player_label(pick)} grades as {_confidence_label(score).lower()} at {score:.1f} on the reliability blend.",
    }


def build_selection_brief(picks: list[dict]) -> dict:
    items = [build_pick_brief(pick) for pick in picks[:5]]
    by_rank = {_rank(pick): pick for pick in picks[:5]}
    ranked_items = sorted(items, key=lambda item: (-item["selection_score"], item["rank"]))
    single = ranked_items[0] if ranked_items else None
    pair = ranked_items[:2]
    pair_players = [item["player"] for item in pair]

    def end_sentence(text: str) -> str:
        return text if text.endswith((".", "!", "?")) else f"{text}."

    if single and len(pair) >= 2:
        single_text = end_sentence(f"Recommended single: {single['player']}")
        pair_text = end_sentence(f"Best two-pick card: {pair_players[0]} and {pair_players[1]}")
        headline = (
            f"{single_text} "
            f"{pair_text}"
        )
    elif single:
        headline = end_sentence(f"Recommended single: {single['player']}")
    else:
        headline = "No learned shortlist picks available."

    for item in items:
        if single and item["rank"] == single["rank"]:
            item["recommendation"] = "Best 1"
        elif item["rank"] in {candidate["rank"] for candidate in pair}:
            item["recommendation"] = "Best 2"
        else:
            item["recommendation"] = "Watch"
        source = by_rank.get(item["rank"], {})
        item["h2h_record"] = source.get("h2h_record") or "0-0"
        item["safety_score"] = source.get("safety_score")
        item["bob_score"] = source.get("bob_score", source.get("score"))
        item["expected_pa"] = source.get("expected_pa")
        item["lineup_slot"] = source.get("lineup_slot")
        item["hitter_ba_season"] = source.get("hitter_ba_season")
        item["probable_pitcher"] = source.get("probable_pitcher")
        item["pickable"] = source.get("pickable")
        item["stop_valve_count"] = source.get("stop_valve_count")
        item["hard_pass_reasons"] = source.get("hard_pass_reasons") or []
        item["concerns"] = source.get("concerns") or []
        item["result_hit"] = source.get("result_hit")
        item["result_hits"] = source.get("result_hits")
        item["result_ab"] = source.get("result_ab")
        item["result_pa"] = source.get("result_pa")
        item["result_status"] = source.get("result_status")
        item["game_state"] = source.get("game_state")
        item["game_status"] = source.get("game_status")
        item["game_hits"] = source.get("game_hits")

    return {
        "policy_version": POLICY_VERSION,
        "headline": headline,
        "recommended_single": single,
        "recommended_pair": pair,
        "items": items,
    }


def _is_evaluable(pick: dict) -> bool:
    status = str(pick.get("result_status") or "").strip().lower()
    return status == "final"


def _is_hit(pick: dict) -> bool:
    if pick.get("result_hit") is True:
        return True
    value = str(pick.get("result_hit") or "").strip()
    if value == "1":
        return True
    return _int(pick.get("result_hits")) > 0


def _timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def prediction_is_pregame(prediction: dict, candidate: dict) -> bool:
    trained = _timestamp(prediction.get("model_trained_at"))
    first_pitch = _timestamp(candidate.get("game_start_time_utc"))
    if trained is None:
        return False
    if first_pitch is not None:
        return trained <= first_pitch
    game_date = str(candidate.get("date") or prediction.get("date") or "").strip()
    return bool(game_date) and trained.date().isoformat() <= game_date


def _longest_streak(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def backtest_pick_payloads(picks_by_date: dict[str, list[dict]]) -> dict:
    daily = []
    top1_results: list[bool] = []
    top2_any_results: list[bool] = []
    top2_both_results: list[bool] = []

    for date_key in sorted(picks_by_date):
        picks = sorted(picks_by_date[date_key], key=_rank)[:5]
        if not picks:
            continue
        brief = build_selection_brief(picks)
        pair_ranks = [item["rank"] for item in brief.get("recommended_pair") or []]
        selected = [pick for pick in picks if _rank(pick) in set(pair_ranks)]
        single_rank = (brief.get("recommended_single") or {}).get("rank")
        single_pick = next((pick for pick in picks if _rank(pick) == single_rank), None)
        if single_pick is None or not selected:
            continue
        if not _is_evaluable(single_pick) or any(not _is_evaluable(pick) for pick in selected):
            continue

        single_hit = _is_hit(single_pick)
        pair_any_hit = any(_is_hit(pick) for pick in selected)
        pair_both_hit = len(selected) >= 2 and all(_is_hit(pick) for pick in selected[:2])
        top1_results.append(single_hit)
        top2_any_results.append(pair_any_hit)
        top2_both_results.append(pair_both_hit)
        daily.append(
            {
                "date": date_key,
                "single_pick": single_pick.get("player"),
                "single_hit": single_hit,
                "pair": [pick.get("player") for pick in selected[:2]],
                "pair_any_hit": pair_any_hit,
                "pair_both_hit": pair_both_hit,
            }
        )

    days = len(daily)
    return {
        "policy_version": POLICY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "top1_hit_rate": round(sum(top1_results) / days, 4) if days else None,
        "top2_any_hit_rate": round(sum(top2_any_results) / days, 4) if days else None,
        "top2_both_hit_rate": round(sum(top2_both_results) / days, 4) if days else None,
        "top1_longest_streak": _longest_streak(top1_results),
        "top2_any_longest_streak": _longest_streak(top2_any_results),
        "daily": daily,
    }


def run_backtest(
    *,
    predictions_csv: Path = DEFAULT_PREDICTIONS_CSV,
    candidates_csv: Path = DEFAULT_OUTPUT_CSV,
    out_json: Path = DEFAULT_SELECTION_BACKTEST_JSON,
    limit: int = 5,
) -> dict:
    from .export_learned_web import (
        build_pick_payload,
        candidate_lookup,
        load_rows,
        rows_by_date,
        sort_predictions,
    )
    from .export_web import row_identity

    prediction_rows = load_rows(predictions_csv)
    candidate_rows = load_rows(candidates_csv)
    candidates_by_date = rows_by_date(candidate_rows)
    predictions_by_date = rows_by_date(prediction_rows)
    picks_by_date: dict[str, list[dict]] = {}

    for date_key, rows in predictions_by_date.items():
        selected_candidates = candidates_by_date.get(date_key, [])
        lookup = candidate_lookup(selected_candidates)
        eligible_predictions = []
        for prediction in rows:
            candidate = lookup.get(row_identity(prediction))
            if candidate and prediction_is_pregame(prediction, candidate):
                eligible_predictions.append(prediction)
        picks_by_date[date_key] = [
            build_pick_payload(prediction, lookup, rank)
            for rank, prediction in enumerate(sort_predictions(eligible_predictions)[:limit], start=1)
        ]

    payload = backtest_pick_payloads(picks_by_date)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest learned top-5 selection policies.")
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV))
    parser.add_argument("--candidates-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--out-json", default=str(DEFAULT_SELECTION_BACKTEST_JSON))
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_backtest(
        predictions_csv=Path(args.predictions_csv),
        candidates_csv=Path(args.candidates_csv),
        out_json=Path(args.out_json),
        limit=args.limit,
    )
    print(
        f"Backtested {payload['days']} learned shortlist day(s): "
        f"top-1 hit rate {payload['top1_hit_rate']}, "
        f"top-2 any-hit rate {payload['top2_any_hit_rate']}."
    )


if __name__ == "__main__":
    main()
