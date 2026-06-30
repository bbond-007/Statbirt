from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .config import DEFAULT_OUTPUT_CSV
from .export_web import (
    DEFAULT_CONGREGATION_CSV,
    available_dates,
    build_game_state_lookup,
    candidate_payload,
    float_value,
    load_congregation,
    load_rows,
    row_identity,
    split_reasons,
    write_json,
)
from .injuries import current_injured_player_ids_for_rows, filter_injured_rows
from .learned_model import DEFAULT_PREDICTIONS_CSV
from .learned_selection import build_selection_brief
from .results import RESULT_STATUS_FINAL, RESULT_STATUS_POSTPONED, normalize_row_result_status
from .utils import parse_float, parse_int

DEFAULT_WEB_DATA_DIR = Path(__file__).resolve().parents[1] / "web" / "data"
DEFAULT_LEARNED_WEB_JSON = DEFAULT_WEB_DATA_DIR / "learned_shortlist.json"
DEFAULT_LEARNED_DASHBOARD_INDEX = DEFAULT_WEB_DATA_DIR / "learned_dashboard_index.json"
DEFAULT_LEARNED_ARCHIVE_DIR = DEFAULT_WEB_DATA_DIR / "learned_dashboards"
DEFAULT_LEARNED_TOP2_THESIS_DIR = Path(__file__).resolve().parents[1] / "data" / "manual" / "learned_top2_theses"
RESULT_FIELDS = frozenset({"result_hit", "result_hits", "result_ab", "result_pa", "result_status"})


def prediction_probability(row: dict[str, str]) -> float:
    return parse_float(row.get("learned_hit_probability")) or 0.0


def prediction_rank(row: dict[str, str]) -> int:
    return parse_int(row.get("learned_rank")) or 999999


def candidate_lookup(rows: list[dict[str, str]]) -> dict[tuple, dict[str, str]]:
    return {row_identity(row): row for row in rows}


def rows_by_date(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        row_date = row.get("date") or ""
        if not row_date:
            continue
        grouped.setdefault(row_date, []).append(row)
    return grouped


def load_top2_thesis(target_date: str, thesis_dir: Path = DEFAULT_LEARNED_TOP2_THESIS_DIR) -> dict | None:
    if not target_date:
        return None
    path = thesis_dir / f"{target_date}.json"
    if not path.exists():
        return None
    try:
        thesis = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse learned top-2 thesis file {path}: {exc}") from exc
    if not isinstance(thesis, dict):
        raise ValueError(f"Learned top-2 thesis file {path} must contain a JSON object.")
    return thesis


def result_hit_value(row: dict[str, str]) -> bool | None:
    value = str(row.get("result_hit") or "").strip()
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def result_int(row: dict[str, str], field: str) -> int | None:
    value = parse_int(row.get(field))
    return value if value is not None else None


def value_present(value: str | None) -> bool:
    return bool(str(value or "").strip())


def merge_prediction_with_candidate(prediction: dict[str, str], candidate: dict[str, str]) -> dict[str, str]:
    merged = dict(candidate)
    for key, value in prediction.items():
        if key in RESULT_FIELDS and value_present(candidate.get(key)):
            continue
        merged[key] = value
    return merged


def apply_game_state_result(payload: dict) -> None:
    if payload.get("result_status") or payload.get("result_hit") is not None:
        return
    game_state = str(payload.get("game_state") or "").lower()
    game_status = str(payload.get("game_status") or "").lower()
    is_final = any(token in game_status for token in ("final", "game over", "completed"))
    if game_state == "postponed":
        payload["result_status"] = RESULT_STATUS_POSTPONED
    elif game_state == "final_no_hit":
        payload["result_status"] = RESULT_STATUS_FINAL
        payload["result_hit"] = False
        payload["result_hits"] = 0
    elif game_state == "hit" and is_final:
        payload["result_status"] = RESULT_STATUS_FINAL
        payload["result_hit"] = True
        payload["result_hits"] = payload.get("game_hits")


def needs_game_state_lookup(rows: list[dict[str, str]]) -> bool:
    return any(normalize_row_result_status(row) in {"", "pending"} for row in rows)


def _flag_bool(row: dict[str, str], field: str) -> bool:
    return str(row.get(field) or "").strip().upper() == "Y"


def safety_profile(row: dict[str, str]) -> dict:
    signals: list[str] = []
    risks: list[str] = []
    score = 50

    expected_pa = float_value(row, "expected_pa")
    if expected_pa is None:
        risks.append("Missing expected PA")
    elif expected_pa >= 4.2:
        score += 12
        signals.append(f"Expected PA {expected_pa:.1f}")
    elif expected_pa >= 4.0:
        score += 5
        signals.append(f"Expected PA {expected_pa:.1f}")
    else:
        score -= 10
        risks.append(f"Expected PA {expected_pa:.1f}")

    pa_per_game = float_value(row, "hitter_pa_per_game_season")
    if pa_per_game is None:
        risks.append("Missing season PA/G")
    elif pa_per_game >= 4.2:
        score += 12
        signals.append(f"Season PA/G {pa_per_game:.2f}")
    elif pa_per_game >= 4.0:
        score += 4
        signals.append(f"Season PA/G {pa_per_game:.2f}")
    else:
        score -= 12
        risks.append(f"Season PA/G {pa_per_game:.2f}")

    lineup_slot = float_value(row, "lineup_slot")
    if lineup_slot is None:
        risks.append("Missing lineup slot")
    elif lineup_slot <= 5:
        score += 8
        signals.append(f"Lineup slot {lineup_slot:.0f}")
    elif lineup_slot <= 7:
        score += 3
        signals.append(f"Lineup slot {lineup_slot:.0f}")
    else:
        score -= 6
        risks.append(f"Lineup slot {lineup_slot:.0f}")

    stop_count = len(split_reasons(row.get("hard_pass_reasons")))
    if stop_count == 0:
        score += 10
        signals.append("No stop valves")
    elif stop_count <= 3:
        score += 4
        signals.append(f"{stop_count} stop valve{'s' if stop_count != 1 else ''}")
    elif stop_count <= 7:
        score -= 4
        risks.append(f"{stop_count} stop valves")
    else:
        score -= 10
        risks.append(f"{stop_count} stop valves")

    k_rate = float_value(row, "hitter_k_rate_500_pa")
    if k_rate is None:
        k_rate = float_value(row, "hitter_k_rate_season")
    if k_rate is None:
        risks.append("Missing K rate")
    elif k_rate <= 0.20:
        score += 8
        signals.append(f"K rate {k_rate * 100:.1f}%")
    elif k_rate <= 0.22:
        score += 4
        signals.append(f"K rate {k_rate * 100:.1f}%")
    else:
        score -= 8
        risks.append(f"K rate {k_rate * 100:.1f}%")

    recent_hipa = float_value(row, "hitter_hipa_500_pa")
    if recent_hipa is not None and recent_hipa >= 0.270:
        score += 5
        signals.append(f"HiPA 500 {recent_hipa:.3f}")
    elif recent_hipa is not None:
        score -= 4
        risks.append(f"HiPA 500 {recent_hipa:.3f}")

    pitcher_stuff = float_value(row, "pitcher_stuff_plus")
    if pitcher_stuff is not None and pitcher_stuff <= 95:
        score += 4
        signals.append(f"Stuff+ {pitcher_stuff:.1f}")
    elif pitcher_stuff is not None and pitcher_stuff > 100:
        score -= 4
        risks.append(f"Stuff+ {pitcher_stuff:.1f}")

    if _flag_bool(row, "confirmed_lineup"):
        score += 3
        signals.append("Confirmed lineup")

    return {
        "safety_score": max(0, min(100, int(round(score)))),
        "selection_signals": signals[:5],
        "selection_risks": risks[:5],
        "stop_valve_count": stop_count,
    }


def sort_predictions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            prediction_rank(row),
            -prediction_probability(row),
            row.get("player") or "",
        ),
    )


def build_pick_payload(
    prediction: dict[str, str],
    candidate_rows: dict[tuple, dict[str, str]],
    rank: int,
    congregation: dict[str, dict] | None = None,
    game_states: dict[tuple[int, int], dict] | None = None,
) -> dict:
    candidate = candidate_rows.get(row_identity(prediction), {})
    merged = merge_prediction_with_candidate(prediction, candidate)
    if not merged.get("score"):
        merged["score"] = prediction.get("bob_score") or candidate.get("score") or ""

    learned_rank = parse_int(prediction.get("learned_rank")) or rank
    payload = candidate_payload(merged, learned_rank, game_states=game_states, congregation=congregation)
    payload.update(
        {
            "rank": learned_rank,
            "learned_rank": learned_rank,
            "learned_hit_probability": round(prediction_probability(prediction), 4),
            "model_version": prediction.get("model_version") or "",
            "model_trained_at": prediction.get("model_trained_at") or "",
            "bob_score": round(parse_float(prediction.get("bob_score")) or float_value(merged, "score") or 0.0, 2),
            "result_hit": result_hit_value(merged),
            "result_hits": result_int(merged, "result_hits"),
            "result_ab": result_int(merged, "result_ab"),
            "result_pa": result_int(merged, "result_pa"),
            "result_status": normalize_row_result_status(merged),
            "matched_candidate": bool(candidate),
        }
    )
    apply_game_state_result(payload)
    payload.update(safety_profile(merged))
    return payload


def learned_summary(picks: list[dict]) -> dict:
    decided = [
        pick for pick in picks
        if pick.get("result_status") == "final" and pick.get("result_hit") is not None
    ]
    hits = [pick for pick in decided if pick.get("result_hit") is True]
    return {
        "top_5_decided_count": len(decided),
        "top_5_hit_count": len(hits),
        "top_5_contains_hit": any(pick.get("result_hit") is True for pick in picks),
    }


def update_learned_dashboard_index(
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
                "path": f"learned_dashboards/{path.name}",
                "generated_at": payload.get("generated_at") or "",
                "total_predictions": payload.get("total_predictions", 0),
                "showing": payload.get("showing") or "",
                "model_version": payload.get("model_version") or "",
                "model_trained_at": payload.get("model_trained_at") or "",
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


def export_learned_web_payload(
    *,
    predictions_csv: Path = DEFAULT_PREDICTIONS_CSV,
    candidates_csv: Path = DEFAULT_OUTPUT_CSV,
    out_json: Path = DEFAULT_LEARNED_WEB_JSON,
    index_json: Path = DEFAULT_LEARNED_DASHBOARD_INDEX,
    archive_dir: Path = DEFAULT_LEARNED_ARCHIVE_DIR,
    congregation_csv: Path = DEFAULT_CONGREGATION_CSV,
    target_date: str | None = None,
    limit: int = 5,
    fallback_latest: bool = True,
    archive: bool = True,
    update_index: bool = True,
    filter_injured: bool = True,
    injured_player_ids: set[int] | frozenset[int] | None = None,
    top2_thesis_dir: Path = DEFAULT_LEARNED_TOP2_THESIS_DIR,
) -> dict:
    predictions = load_rows(predictions_csv)
    candidates = load_rows(candidates_csv)
    if filter_injured and injured_player_ids is None:
        injured_player_ids = current_injured_player_ids_for_rows(candidates)
    congregation = load_congregation(congregation_csv)
    return export_learned_web_payload_from_rows(
        predictions_by_date=rows_by_date(predictions),
        candidates_by_date=rows_by_date(candidates),
        out_json=out_json,
        index_json=index_json,
        archive_dir=archive_dir,
        target_date=target_date,
        limit=limit,
        fallback_latest=fallback_latest,
        archive=archive,
        update_index=update_index,
        filter_injured=filter_injured,
        injured_player_ids=injured_player_ids,
        congregation=congregation,
        top2_thesis_dir=top2_thesis_dir,
    )


def export_learned_web_payload_from_rows(
    *,
    predictions_by_date: dict[str, list[dict[str, str]]],
    candidates_by_date: dict[str, list[dict[str, str]]],
    out_json: Path = DEFAULT_LEARNED_WEB_JSON,
    index_json: Path = DEFAULT_LEARNED_DASHBOARD_INDEX,
    archive_dir: Path = DEFAULT_LEARNED_ARCHIVE_DIR,
    target_date: str | None = None,
    limit: int = 5,
    fallback_latest: bool = True,
    archive: bool = True,
    update_index: bool = True,
    filter_injured: bool = True,
    injured_player_ids: set[int] | frozenset[int] | None = None,
    congregation: dict[str, dict] | None = None,
    top2_thesis_dir: Path = DEFAULT_LEARNED_TOP2_THESIS_DIR,
) -> dict:
    dates = sorted(predictions_by_date)
    if target_date == "latest":
        requested_date = dates[-1] if dates else date.today().isoformat()
    else:
        requested_date = target_date or date.today().isoformat()
    selected = list(predictions_by_date.get(requested_date, []))
    selected_date = requested_date
    if not selected and fallback_latest and dates:
        selected_date = dates[-1]
        selected = list(predictions_by_date.get(selected_date, []))

    if filter_injured:
        selected, injury_filtered_count = filter_injured_rows(selected, injured_player_ids or set())
    else:
        injury_filtered_count = 0

    selected = sort_predictions(selected)
    selected_candidates = candidates_by_date.get(selected_date, [])
    if filter_injured:
        selected_candidates, _ = filter_injured_rows(selected_candidates, injured_player_ids or set())
    lookup = candidate_lookup(selected_candidates)
    display_predictions = selected[:limit]
    game_state_rows = [
        merge_prediction_with_candidate(prediction, lookup.get(row_identity(prediction), {}))
        for prediction in display_predictions
    ]
    selected_day = date.fromisoformat(selected_date) if selected_date else None
    game_states = build_game_state_lookup(selected_day, game_state_rows) if needs_game_state_lookup(game_state_rows) else {}
    picks = [
        build_pick_payload(prediction, lookup, rank, congregation, game_states)
        for rank, prediction in enumerate(display_predictions, start=1)
    ]
    model_version = next((pick.get("model_version") for pick in picks if pick.get("model_version")), "")
    model_trained_at = next((pick.get("model_trained_at") for pick in picks if pick.get("model_trained_at")), "")
    summary = learned_summary(picks)
    top2_thesis = load_top2_thesis(selected_date, top2_thesis_dir)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_date": requested_date,
        "date": selected_date,
        "used_latest_fallback": selected_date != requested_date,
        "total_predictions": len(selected),
        "total_candidates": len(selected_candidates),
        "injury_filtered_count": injury_filtered_count,
        "showing": "learned_model_shortlist",
        "limit": limit,
        "model_version": model_version,
        "model_trained_at": model_trained_at,
        "daily_selection_brief": build_selection_brief(picks),
        **summary,
        "picks": picks,
    }
    if top2_thesis:
        payload["learned_top2_thesis"] = top2_thesis
    write_json(out_json, payload)
    if archive and payload.get("date"):
        archive_path = archive_dir / f"{payload['date']}.json"
        write_json(archive_path, payload)
        if update_index:
            update_learned_dashboard_index(active_payload=payload, archive_dir=archive_dir, index_json=index_json)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Export learned-model shortlist JSON for the web dashboard.")
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV))
    parser.add_argument("--candidates-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--out-json", default=str(DEFAULT_LEARNED_WEB_JSON))
    parser.add_argument("--index-json", default=str(DEFAULT_LEARNED_DASHBOARD_INDEX))
    parser.add_argument("--archive-dir", default=str(DEFAULT_LEARNED_ARCHIVE_DIR))
    parser.add_argument("--congregation-csv", default=str(DEFAULT_CONGREGATION_CSV))
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD or latest.")
    parser.add_argument("--all-dates", action="store_true", help="Export one learned dashboard JSON for every prediction date.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top2-thesis-dir", default=str(DEFAULT_LEARNED_TOP2_THESIS_DIR))
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--no-fallback-latest", action="store_true")
    parser.add_argument("--include-injured", action="store_true", help="Do not filter players currently listed as injured.")
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_csv = Path(args.predictions_csv)
    candidates_csv = Path(args.candidates_csv)
    prediction_rows = load_rows(predictions_csv)
    candidate_rows = load_rows(candidates_csv)
    injured_player_ids = None
    if not args.include_injured:
        injured_player_ids = current_injured_player_ids_for_rows(candidate_rows)
    prediction_rows_by_date = rows_by_date(prediction_rows)
    candidate_rows_by_date = rows_by_date(candidate_rows)
    congregation = load_congregation(Path(args.congregation_csv))
    target_dates = [args.date]
    if args.all_dates:
        target_dates = available_dates(prediction_rows)

    payloads = []
    for target_date in target_dates:
        payloads.append(
            export_learned_web_payload_from_rows(
                predictions_by_date=prediction_rows_by_date,
                candidates_by_date=candidate_rows_by_date,
                out_json=Path(args.out_json),
                index_json=Path(args.index_json),
                archive_dir=Path(args.archive_dir),
                target_date=target_date,
                limit=args.limit,
                fallback_latest=not args.no_fallback_latest,
                archive=not args.no_archive,
                update_index=not args.all_dates,
                filter_injured=not args.include_injured,
                injured_player_ids=injured_player_ids,
                congregation=congregation,
                top2_thesis_dir=Path(args.top2_thesis_dir),
            )
        )
    if args.all_dates and payloads and not args.no_archive:
        update_learned_dashboard_index(
            active_payload=payloads[-1],
            archive_dir=Path(args.archive_dir),
            index_json=Path(args.index_json),
        )
    latest = payloads[-1] if payloads else {"picks": [], "date": ""}
    print(
        f"Exported {len(payloads)} learned dashboard file(s); active board has "
        f"{len(latest['picks'])} learned pick(s) for {latest['date']} at {Path(args.out_json).resolve()}."
    )


if __name__ == "__main__":
    main()
