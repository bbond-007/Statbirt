from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .config import DATA_DIR, DEFAULT_OUTPUT_CSV
from .learned_model import DEFAULT_PREDICTIONS_CSV, load_rows
from .prediction_ledger import DEFAULT_RESULTS_CSV, DEFAULT_SNAPSHOTS_CSV
from .results import RESULT_STATUS_FINAL, RESULT_STATUS_NO_APPEARANCE, normalize_row_result_status
from .utils import parse_float, parse_int


DEFAULT_SHADOW_MODEL_JSON = DATA_DIR / "models" / "learned_shadow_model.json"
DEFAULT_SHADOW_REPORT_JSON = DATA_DIR / "models" / "learned_shadow_report.json"
DEFAULT_SHADOW_PREDICTIONS_CSV = DATA_DIR / "learned_shadow_predictions.csv"
DEFAULT_PROMOTION_REPORT_JSON = DATA_DIR / "models" / "learned_shadow_promotion.json"
SHADOW_START_DATE = date(2026, 7, 18)
PROMOTION_MIN_RESOLVED_DAYS = 50
DEPLOYMENT_CALIBRATION_DATES = 14

IDENTITY_FIELDS = ["date", "player", "player_id", "team", "opponent", "game_pk"]

APPEARANCE_FEATURES = [
    "official_lineup",
    "lineup_slot",
    "expected_pa",
    "starts_last_5",
    "hitter_pa_per_game_season",
    "last5_ab_per_game",
    "road_game",
    "doubleheader",
    "active_roster",
    "days_since_activation",
]

HIT_FEATURES = [
    "hitter_ba_500_ab",
    "hitter_hipa_500_pa",
    "hitter_ba_2500_ab",
    "season_ba_shrunk",
    "matchup_hand_ba_shrunk",
    "hitter_whiff_rate_500_pa",
    "hitter_k_rate_500_pa",
    "hitter_whiff_rate_season",
    "hitter_k_rate_season",
    "last5_ab_per_game",
    "pitcher_stuff_plus",
    "pitcher_hpi_200",
    "pitcher_hpi_season",
    "pitcher_hits_last_18_ip",
    "inferred_pitch_type_xba",
    "inferred_pitch_type_contact_rate",
    "inferred_pitch_type_shape_distance",
    "h2h_quality_shrunk",
    "bullpen_hpi",
    "bullpen_opp_ba",
    "park_hit_factor",
    "forecast_temperature_f",
    "stop_valve_count",
    # These prospective fields remain median-imputed until collection begins.
    "hitter_xba_100_pa",
    "hitter_hard_hit_rate_50_bbe",
    "hitter_sweet_spot_rate_50_bbe",
    "hitter_ev50_50_bbe",
    "hitter_bat_speed_50_swings",
    "hitter_swing_length_50_swings",
    "hitter_competitive_contact_rate_season",
    "hitter_avg_bat_speed_season",
    "hitter_avg_swing_length_season",
    "hitter_squared_up_per_contact_season",
    "hitter_blast_per_contact_season",
    "pitcher_xba_allowed_100_bf",
    "pitcher_hard_hit_rate_allowed_50_bbe",
    "bullpen_pitches_3d",
    "bullpen_relief_appearances_3d",
    "opponent_team_oaa",
    "opponent_infield_oaa",
    "opponent_outfield_oaa",
]

SHADOW_PREDICTION_FIELDS = [
    *IDENTITY_FIELDS,
    "candidate_pool_source",
    "lineup_source",
    "production_rank",
    "production_probability",
    "production_model_version",
    "production_model_trained_at",
    "appearance_probability",
    "hit_given_appearance_probability",
    "combined_probability_raw",
    "combined_probability_calibrated",
    "shadow_rank",
    "shadow_top5_rank",
    "stuff_preference_rank",
    "stop_valve_count",
    "model_version",
    "model_trained_at",
    "scored_at",
    "feature_snapshot_id",
    "feature_source_hash",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _date_text(row: dict) -> str:
    return str(row.get("date") or row.get("target_date") or "").strip()


def _parse_date(value) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return None


def _is_yes(value) -> bool:
    return str(value or "").strip().lower() in {"y", "yes", "true", "1"}


def _split_reasons(value) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _key(row: dict) -> tuple[str, str, str]:
    return (_date_text(row), str(row.get("player_id") or ""), str(row.get("game_pk") or ""))


def _number(row: dict, field: str) -> float | None:
    return parse_float(row.get(field))


def _rank(row: dict, field: str) -> int:
    return parse_int(row.get(field)) or 999999


def _lineup_source(row: dict) -> str:
    source = str(row.get("lineup_source") or "").strip().lower()
    if source:
        return source
    # Legacy rows do not reveal whether Y came from a morning lineup or final boxscore.
    return "unknown"


def _official_lineup(row: dict) -> float:
    return 1.0 if _lineup_source(row) == "official" and _is_yes(row.get("confirmed_lineup")) else 0.0


def _shrink(rate: float | None, sample: float | None, prior: float, strength: float) -> float:
    if rate is None or sample is None or sample <= 0:
        return prior
    return (rate * sample + prior * strength) / (sample + strength)


def derived_features(row: dict) -> dict[str, float | None]:
    last5_games = _number(row, "hitter_last_5_games_played")
    last5_ab = _number(row, "hitter_last_5_games_ab")
    last5_ab_per_game = None
    if last5_games is not None and last5_games > 0 and last5_ab is not None:
        last5_ab_per_game = last5_ab / last5_games

    season_pa_l = _number(row, "hitter_split_pa_season_vs_lhp") or 0.0
    season_pa_r = _number(row, "hitter_split_pa_season_vs_rhp") or 0.0
    season_pa = season_pa_l + season_pa_r
    season_ba = _number(row, "hitter_ba_season")
    season_ba_shrunk = _shrink(season_ba, season_pa, 0.250, 100.0)

    pitcher_hand = str(row.get("pitcher_hand") or "").strip().upper()
    hand_suffix = "lhp" if pitcher_hand == "L" else "rhp"
    hand_rate = _number(row, f"hitter_split_ba_season_vs_{hand_suffix}")
    hand_pa = _number(row, f"hitter_split_pa_season_vs_{hand_suffix}")
    hand_prior = _number(row, f"hitter_split_ba_500_vs_{hand_suffix}") or season_ba_shrunk
    hand_shrunk = _shrink(hand_rate, hand_pa, hand_prior, 50.0)

    pitch_type_prior = _number(row, "inferred_pitch_type_xba")
    if pitch_type_prior is None:
        pitch_type_prior = hand_shrunk
    h2h_pa = _number(row, "h2h_pa") or 0.0
    h2h_quality = _number(row, "h2h_xba")
    if h2h_quality is None:
        h2h_quality = _number(row, "h2h_hit_rate")
    h2h_quality_shrunk = _shrink(h2h_quality, h2h_pa, pitch_type_prior, 25.0)

    output: dict[str, float | None] = {
        field: _number(row, field)
        for field in set(APPEARANCE_FEATURES + HIT_FEATURES)
        if field not in {
            "official_lineup",
            "last5_ab_per_game",
            "season_ba_shrunk",
            "matchup_hand_ba_shrunk",
            "h2h_quality_shrunk",
            "stop_valve_count",
            "road_game",
            "doubleheader",
            "active_roster",
        }
    }
    output.update(
        {
            "official_lineup": _official_lineup(row),
            "last5_ab_per_game": last5_ab_per_game,
            "season_ba_shrunk": season_ba_shrunk,
            "matchup_hand_ba_shrunk": hand_shrunk,
            "h2h_quality_shrunk": h2h_quality_shrunk,
            "stop_valve_count": float(len(_split_reasons(row.get("hard_pass_reasons")))),
            "road_game": 1.0 if _is_yes(row.get("road_game")) else 0.0,
            "doubleheader": 1.0 if _is_yes(row.get("doubleheader")) else 0.0,
            "active_roster": (
                1.0
                if _is_yes(row.get("active_roster"))
                else 0.0
                if str(row.get("active_roster") or "").strip()
                else None
            ),
        }
    )
    return output


def appearance_label(row: dict) -> int | None:
    source = _lineup_source(row)
    if source == "final_boxscore" or (source == "unknown" and _is_yes(row.get("confirmed_lineup"))):
        return None
    status = normalize_row_result_status(row)
    if status == RESULT_STATUS_FINAL:
        return 1
    if status == RESULT_STATUS_NO_APPEARANCE:
        return 0
    return None


def hit_label(row: dict) -> int | None:
    if normalize_row_result_status(row) != RESULT_STATUS_FINAL:
        return None
    value = str(row.get("result_hit") or "").strip()
    if value in {"0", "1"}:
        return int(value)
    return None


def decision_outcome(row: dict) -> int | None:
    status = normalize_row_result_status(row)
    if status == RESULT_STATUS_NO_APPEARANCE:
        return 0
    return hit_label(row)


def _matrix_fit(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    values = [derived_features(row) for row in rows]
    medians: dict[str, float] = {}
    for feature in feature_names:
        observed = [float(item[feature]) for item in values if item.get(feature) is not None]
        medians[feature] = float(np.median(observed)) if observed else 0.0
    matrix = np.array(
        [[medians[name] if item.get(name) is None else float(item[name]) for name in feature_names] for item in values],
        dtype=float,
    )
    return matrix, medians


def _matrix_apply(rows: list[dict], model: dict) -> np.ndarray:
    names = model["feature_names"]
    medians = model["medians"]
    values = [derived_features(row) for row in rows]
    return np.array(
        [[float(medians[name]) if item.get(name) is None else float(item[name]) for name in names] for item in values],
        dtype=float,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _recency_weights(rows: list[dict], half_life_days: float) -> np.ndarray:
    parsed = [_parse_date(_date_text(row)) for row in rows]
    newest = max((value for value in parsed if value is not None), default=date.today())
    weights = []
    for value in parsed:
        age = max(0, (newest - value).days) if value else 0
        weights.append(max(0.08, 0.5 ** (age / half_life_days)))
    return np.array(weights, dtype=float)


def _fit_component(
    rows: list[dict],
    labels: list[int],
    feature_names: list[str],
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
    half_life_days: float,
) -> dict:
    x, medians = _matrix_fit(rows, feature_names)
    y = np.array(labels, dtype=float)
    means = x.mean(axis=0)
    stds = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
    x_std = (x - means) / stds
    sample_weights = _recency_weights(rows, half_life_days)
    weight_sum = float(sample_weights.sum())
    prevalence = min(max(float(np.average(y, weights=sample_weights)), 1e-4), 1 - 1e-4)
    intercept = math.log(prevalence / (1 - prevalence))
    weights = np.zeros(x_std.shape[1], dtype=float)
    for step in range(iterations):
        probabilities = _sigmoid(x_std @ weights + intercept)
        error = (probabilities - y) * sample_weights
        gradient = (x_std.T @ error) / weight_sum + l2 * weights
        intercept_gradient = float(error.sum() / weight_sum)
        rate = learning_rate / math.sqrt(1.0 + step / 500.0)
        weights -= rate * gradient
        intercept -= rate * intercept_gradient
    return {
        "feature_names": feature_names,
        "medians": medians,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "intercept": float(intercept),
        "training_rows": len(rows),
        "weighted_prevalence": prevalence,
    }


def _predict_component(model: dict, rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    x = _matrix_apply(rows, model)
    means = np.array(model["means"], dtype=float)
    stds = np.array(model["stds"], dtype=float)
    weights = np.array(model["weights"], dtype=float)
    return _sigmoid(((x - means) / stds) @ weights + float(model["intercept"]))


def _fit_platt(raw: np.ndarray, labels: np.ndarray) -> dict:
    if len(raw) < 30 or len(set(labels.astype(int).tolist())) < 2:
        return {"type": "identity", "slope": 1.0, "intercept": 0.0, "rows": len(raw)}
    logits = np.log(np.clip(raw, 1e-5, 1 - 1e-5) / np.clip(1 - raw, 1e-5, 1 - 1e-5))
    slope = 1.0
    intercept = 0.0
    for step in range(600):
        predicted = _sigmoid(logits * slope + intercept)
        error = predicted - labels
        rate = 0.04 / math.sqrt(1.0 + step / 200.0)
        slope -= rate * float(np.mean(error * logits) + 0.01 * (slope - 1.0))
        intercept -= rate * float(np.mean(error))
    return {"type": "platt", "slope": float(slope), "intercept": float(intercept), "rows": len(raw)}


def calibrate(probabilities: np.ndarray, calibrator: dict) -> np.ndarray:
    if calibrator.get("type") != "platt":
        return probabilities.copy()
    logits = np.log(
        np.clip(probabilities, 1e-5, 1 - 1e-5) / np.clip(1 - probabilities, 1e-5, 1 - 1e-5)
    )
    return _sigmoid(logits * float(calibrator["slope"]) + float(calibrator["intercept"]))


def _date_split(rows: list[dict], fraction: float = 0.8) -> tuple[list[dict], list[dict]]:
    dates = sorted({_date_text(row) for row in rows if _date_text(row)})
    if len(dates) < 5:
        return rows, []
    cutoff = max(1, min(len(dates) - 1, int(len(dates) * fraction)))
    train_dates = set(dates[:cutoff])
    return [row for row in rows if _date_text(row) in train_dates], [row for row in rows if _date_text(row) not in train_dates]


def _date_three_way_split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    dates = sorted({_date_text(row) for row in rows if _date_text(row)})
    if len(dates) < 10:
        return rows, [], []
    first_cut = max(1, int(len(dates) * 0.60))
    second_cut = max(first_cut + 1, int(len(dates) * 0.80))
    second_cut = min(second_cut, len(dates) - 1)
    train_dates = set(dates[:first_cut])
    calibration_dates = set(dates[first_cut:second_cut])
    return (
        [row for row in rows if _date_text(row) in train_dates],
        [row for row in rows if _date_text(row) in calibration_dates],
        [row for row in rows if _date_text(row) not in train_dates | calibration_dates],
    )


def _deployment_split(
    rows: list[dict], holdout_dates: int = DEPLOYMENT_CALIBRATION_DATES
) -> tuple[list[dict], list[dict]]:
    resolved_dates = sorted(
        {_date_text(row) for row in rows if _date_text(row) and decision_outcome(row) is not None}
    )
    if len(resolved_dates) <= holdout_dates + 5:
        return rows, []
    calibration_dates = set(resolved_dates[-holdout_dates:])
    calibration_start = min(calibration_dates)
    return (
        [row for row in rows if _date_text(row) < calibration_start],
        [row for row in rows if _date_text(row) in calibration_dates],
    )


def _labeled(rows: list[dict], label_fn) -> tuple[list[dict], list[int]]:
    pairs = [(row, label_fn(row)) for row in rows]
    selected = [(row, label) for row, label in pairs if label is not None]
    return [row for row, _ in selected], [int(label) for _, label in selected]


def _brier(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    return float(np.mean((labels - probabilities) ** 2)) if len(labels) else None


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if not len(labels):
        return None
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def _ranking_validation(
    rows: list[dict],
    appearance_model: dict,
    hit_model: dict,
    calibrator: dict,
    production_rows: list[dict],
) -> dict:
    production = {_key(row): row for row in production_rows}
    top5_rows = [
        row for row in rows if _rank(production.get(_key(row), {}), "learned_rank") <= 5
    ]
    if not top5_rows:
        return {"resolved_top_one_days": 0, "resolved_top_two_days": 0}
    raw = _predict_component(appearance_model, top5_rows) * _predict_component(hit_model, top5_rows)
    probabilities = calibrate(raw, calibrator)
    grouped: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for row, probability in zip(top5_rows, probabilities):
        grouped[_date_text(row)].append((row, float(probability)))

    shadow_top_one: list[int] = []
    shadow_top_two: list[int] = []
    shadow_top_two_both: list[int] = []
    production_top_one: list[int] = []
    production_top_two: list[int] = []
    for day in sorted(grouped):
        if len(grouped[day]) != 5:
            continue
        shadow_ranked = sorted(
            grouped[day],
            key=lambda item: (-item[1], str(item[0].get("player_id") or ""), str(item[0].get("game_pk") or "")),
        )
        production_ranked = sorted(
            grouped[day],
            key=lambda item: (
                _rank(production.get(_key(item[0]), {}), "learned_rank"),
                str(item[0].get("player_id") or ""),
            ),
        )
        if any(decision_outcome(item[0]) is None for item in shadow_ranked):
            continue
        shadow_labels = [int(decision_outcome(item[0])) for item in shadow_ranked]
        production_labels = [int(decision_outcome(item[0])) for item in production_ranked]
        shadow_top_one.append(shadow_labels[0])
        shadow_top_two.append(int(any(shadow_labels[:2])))
        shadow_top_two_both.append(int(all(shadow_labels[:2])))
        production_top_one.append(production_labels[0])
        production_top_two.append(int(any(production_labels[:2])))

    return {
        "date_min": min(grouped),
        "date_max": max(grouped),
        "scope": "rerank_within_saved_production_top_five",
        "resolved_top_one_days": len(shadow_top_one),
        "top_one_hits": sum(shadow_top_one),
        "top_one_hit_rate": (sum(shadow_top_one) / len(shadow_top_one)) if shadow_top_one else None,
        "top_one_longest_miss_streak": _longest_miss_streak(shadow_top_one),
        "resolved_top_two_days": len(shadow_top_two),
        "top_two_any_hit_days": sum(shadow_top_two),
        "top_two_any_hit_rate": (sum(shadow_top_two) / len(shadow_top_two)) if shadow_top_two else None,
        "top_two_both_hit_days": sum(shadow_top_two_both),
        "top_two_both_hit_rate": (
            sum(shadow_top_two_both) / len(shadow_top_two_both) if shadow_top_two_both else None
        ),
        "production_top_one_hits": sum(production_top_one),
        "production_top_one_hit_rate": (
            sum(production_top_one) / len(production_top_one) if production_top_one else None
        ),
        "production_top_two_any_hit_days": sum(production_top_two),
        "production_top_two_any_hit_rate": (
            sum(production_top_two) / len(production_top_two) if production_top_two else None
        ),
    }


def train_shadow_model(
    candidates_csv: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    production_predictions_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    model_out: str | Path = DEFAULT_SHADOW_MODEL_JSON,
    report_out: str | Path = DEFAULT_SHADOW_REPORT_JSON,
    iterations: int = 360,
    learning_rate: float = 0.05,
    l2: float = 0.04,
    half_life_days: float = 180.0,
) -> dict:
    all_rows = load_rows(candidates_csv)
    production_rows = load_rows(production_predictions_csv)
    appearance_rows, appearance_labels = _labeled(all_rows, appearance_label)
    hit_rows, hit_labels = _labeled(all_rows, hit_label)
    if len(appearance_rows) < 200 or len(hit_rows) < 200:
        raise ValueError("Shadow training requires at least 200 appearance and hit rows.")

    calibration_train_rows, calibrator_rows, untouched_evaluation_rows = _date_three_way_split(all_rows)
    evaluation_start = min(
        (_parse_date(_date_text(row)) for row in untouched_evaluation_rows if _parse_date(_date_text(row))),
        default=None,
    )
    appearance_pre_evaluation = [
        row
        for row in appearance_rows
        if evaluation_start is None or (_parse_date(_date_text(row)) or date.max) < evaluation_start
    ]
    app_train_pool, _ = _date_split(appearance_pre_evaluation, fraction=0.70)
    app_train, app_train_y = _labeled(app_train_pool, appearance_label)
    hit_train, hit_train_y = _labeled(calibration_train_rows, hit_label)
    if len(app_train) >= 200:
        app_training_max = max(_date_text(row) for row in app_train)
        calibrator_rows = [row for row in calibrator_rows if _date_text(row) > app_training_max]
    calibrator_fit, calibrator_y = _labeled(calibrator_rows, decision_outcome)
    calibration_eval, calibration_y = _labeled(untouched_evaluation_rows, decision_outcome)

    calibration_model = None
    calibration_metrics = None
    untouched_ranking = None
    if (
        len(app_train) >= 200
        and len(hit_train) >= 200
        and calibrator_fit
        and calibration_eval
        and len(set(calibrator_y)) == 2
        and len(set(calibration_y)) == 2
    ):
        app_split = _fit_component(
            app_train,
            app_train_y,
            APPEARANCE_FEATURES,
            iterations=iterations,
            learning_rate=learning_rate,
            l2=l2,
            half_life_days=half_life_days,
        )
        hit_split = _fit_component(
            hit_train,
            hit_train_y,
            HIT_FEATURES,
            iterations=iterations,
            learning_rate=learning_rate,
            l2=l2,
            half_life_days=half_life_days,
        )
        calibrator_raw = _predict_component(app_split, calibrator_fit) * _predict_component(hit_split, calibrator_fit)
        calibration_model = _fit_platt(calibrator_raw, np.array(calibrator_y, dtype=float))
        raw = _predict_component(app_split, calibration_eval) * _predict_component(hit_split, calibration_eval)
        y_cal = np.array(calibration_y, dtype=float)
        calibrated = calibrate(raw, calibration_model)
        calibration_metrics = {
            "calibrator_rows": len(calibrator_fit),
            "calibrator_date_min": min(_date_text(row) for row in calibrator_fit),
            "calibrator_date_max": max(_date_text(row) for row in calibrator_fit),
            "appearance_training_rows": len(app_train),
            "appearance_training_date_min": min(_date_text(row) for row in app_train),
            "appearance_training_date_max": max(_date_text(row) for row in app_train),
            "hit_training_rows": len(hit_train),
            "hit_training_date_min": min(_date_text(row) for row in hit_train),
            "hit_training_date_max": max(_date_text(row) for row in hit_train),
            "rows": len(calibration_eval),
            "date_min": min(_date_text(row) for row in calibration_eval),
            "date_max": max(_date_text(row) for row in calibration_eval),
            "raw_brier": _brier(y_cal, raw),
            "calibrated_brier": _brier(y_cal, calibrated),
            "raw_log_loss": _log_loss(y_cal, raw),
            "calibrated_log_loss": _log_loss(y_cal, calibrated),
        }
        appearance_eval_rows, appearance_eval_labels = _labeled(
            untouched_evaluation_rows, appearance_label
        )
        if appearance_eval_rows:
            appearance_eval_probability = _predict_component(app_split, appearance_eval_rows)
            appearance_eval_y = np.array(appearance_eval_labels, dtype=float)
            calibration_metrics.update(
                {
                    "appearance_rows": len(appearance_eval_rows),
                    "appearance_brier": _brier(appearance_eval_y, appearance_eval_probability),
                    "appearance_log_loss": _log_loss(appearance_eval_y, appearance_eval_probability),
                }
            )
        untouched_ranking = _ranking_validation(
            untouched_evaluation_rows,
            app_split,
            hit_split,
            calibration_model,
            production_rows,
        )
    if calibration_model is None:
        calibration_model = {"type": "identity", "slope": 1.0, "intercept": 0.0, "rows": 0}

    deployment_train_rows, deployment_calibration_rows = _deployment_split(all_rows)
    deployment_appearance_rows, deployment_appearance_labels = _labeled(
        deployment_train_rows, appearance_label
    )
    deployment_hit_rows, deployment_hit_labels = _labeled(deployment_train_rows, hit_label)
    deployment_calibrator_rows, deployment_calibrator_labels = _labeled(
        deployment_calibration_rows, decision_outcome
    )
    if (
        len(deployment_appearance_rows) < 200
        or len(deployment_hit_rows) < 200
        or len(deployment_calibrator_rows) < 30
    ):
        raise ValueError(
            "Shadow deployment split requires at least 200 appearance/hit rows and 30 calibrator rows."
        )
    appearance_model = _fit_component(
        deployment_appearance_rows,
        deployment_appearance_labels,
        APPEARANCE_FEATURES,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
        half_life_days=half_life_days,
    )
    hit_model = _fit_component(
        deployment_hit_rows,
        deployment_hit_labels,
        HIT_FEATURES,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
        half_life_days=half_life_days,
    )
    deployment_raw = _predict_component(appearance_model, deployment_calibrator_rows) * _predict_component(
        hit_model, deployment_calibrator_rows
    )
    deployment_calibrator = _fit_platt(
        deployment_raw,
        np.array(deployment_calibrator_labels, dtype=float),
    )
    trained_at = _now_utc()
    model = {
        "model_type": "two_stage_recency_weighted_logistic_with_platt_calibration",
        "model_version": f"learned-shadow-v1-{trained_at.replace(':', '').replace('-', '')}",
        "trained_at": trained_at,
        "shadow_started_on": SHADOW_START_DATE.isoformat(),
        "production_model_unchanged": True,
        "parameters": {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "l2": l2,
            "half_life_days": half_life_days,
            "deployment_calibration_holdout_dates": DEPLOYMENT_CALIBRATION_DATES,
            "calendar_fields": "excluded",
            "raw_rain": "excluded",
            "bob_score": "excluded",
            "missingness_indicators": "excluded",
            "h2h": "shrunk_toward_pitch_type_and_hand_prior_strength_25",
            "season_ba": "shrunk_toward_250_with_100_pa_prior",
            "current_hand_ba": "shrunk_toward_500_split_with_50_pa_prior",
        },
        "appearance_model": appearance_model,
        "hit_given_appearance_model": hit_model,
        "calibrator": deployment_calibrator,
    }
    report = {
        "model_version": model["model_version"],
        "trained_at": trained_at,
        "candidate_rows": len(all_rows),
        "appearance_eligible_rows": len(appearance_rows),
        "appearance_training_rows": len(deployment_appearance_rows),
        "appearance_rate": float(np.mean(deployment_appearance_labels)),
        "hit_eligible_rows": len(hit_rows),
        "hit_training_rows": len(deployment_hit_rows),
        "hit_given_appearance_rate": float(np.mean(deployment_hit_labels)),
        "calibration": calibration_metrics,
        "deployment_calibration": {
            "holdout_dates": DEPLOYMENT_CALIBRATION_DATES,
            "training_date_max": max(_date_text(row) for row in deployment_train_rows),
            "calibrator_rows": len(deployment_calibrator_rows),
            "calibrator_date_min": min(_date_text(row) for row in deployment_calibrator_rows),
            "calibrator_date_max": max(_date_text(row) for row in deployment_calibrator_rows),
            "calibrator_type": deployment_calibrator.get("type"),
        },
        "untouched_ranking_validation": untouched_ranking,
        "feature_contract": {
            "appearance": APPEARANCE_FEATURES,
            "hit_given_appearance": HIT_FEATURES,
        },
        "promotion_gate": {
            "shadow_start_date": SHADOW_START_DATE.isoformat(),
            "minimum_resolved_days": PROMOTION_MIN_RESOLVED_DAYS,
            "status": "collecting_prospective_evidence",
        },
    }
    model_path = Path(model_out)
    report_path = Path(report_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"model": model, "report": report, "model_path": model_path, "report_path": report_path}


def load_shadow_model(path: str | Path = DEFAULT_SHADOW_MODEL_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _filter_date(rows: list[dict], date_filter: str) -> list[dict]:
    dates = sorted({_date_text(row) for row in rows if _date_text(row)})
    if not dates:
        return []
    target = dates[-1] if date_filter == "latest" else date_filter
    return [row for row in rows if _date_text(row) == target]


def _production_lookup(path: str | Path, target_date: str) -> dict[tuple[str, str, str], dict]:
    return {_key(row): row for row in load_rows(path) if _date_text(row) == target_date}


def _write_predictions(path: Path, records: list[dict]) -> None:
    existing = load_rows(path)
    keys = {_key(row) for row in records}
    merged = [row for row in existing if _key(row) not in keys]
    merged.extend(records)
    merged.sort(key=lambda row: (_date_text(row), -_rank(row, "shadow_rank")), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHADOW_PREDICTION_FIELDS)
        writer.writeheader()
        for row in merged:
            writer.writerow({field: row.get(field, "") for field in SHADOW_PREDICTION_FIELDS})


def score_shadow_candidates(
    candidates_csv: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    production_predictions_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    model_path: str | Path = DEFAULT_SHADOW_MODEL_JSON,
    out_csv: str | Path = DEFAULT_SHADOW_PREDICTIONS_CSV,
    date_filter: str = "latest",
) -> list[dict]:
    all_candidates = load_rows(candidates_csv)
    rows = _filter_date(all_candidates, date_filter)
    if not rows:
        raise ValueError("No candidates matched the requested shadow score date.")
    target_date = _date_text(rows[0])
    model = load_shadow_model(model_path)
    app_probability = _predict_component(model["appearance_model"], rows)
    hit_probability = _predict_component(model["hit_given_appearance_model"], rows)
    combined_raw = app_probability * hit_probability
    combined_calibrated = calibrate(combined_raw, model["calibrator"])
    production = _production_lookup(production_predictions_csv, target_date)

    order = sorted(
        range(len(rows)),
        key=lambda idx: (
            -float(combined_calibrated[idx]),
            _rank(production.get(_key(rows[idx]), {}), "learned_rank"),
            str(rows[idx].get("player_id") or ""),
        ),
    )
    shadow_ranks = [0] * len(rows)
    for rank, idx in enumerate(order, start=1):
        shadow_ranks[idx] = rank

    top5_indexes = [idx for idx, row in enumerate(rows) if _rank(production.get(_key(row), {}), "learned_rank") <= 5]
    top5_order = sorted(
        top5_indexes,
        key=lambda idx: (
            -float(combined_calibrated[idx]),
            _rank(production.get(_key(rows[idx]), {}), "learned_rank"),
            str(rows[idx].get("player_id") or ""),
        ),
    )
    shadow_top5_ranks = {idx: rank for rank, idx in enumerate(top5_order, start=1)}
    stuff_order = sorted(
        top5_indexes,
        key=lambda idx: (
            0 if (_number(rows[idx], "pitcher_stuff_plus") is not None and _number(rows[idx], "pitcher_stuff_plus") <= 100) else 1,
            -float(combined_calibrated[idx]),
            _rank(production.get(_key(rows[idx]), {}), "learned_rank"),
            str(rows[idx].get("player_id") or ""),
        ),
    )
    stuff_ranks = {idx: rank for rank, idx in enumerate(stuff_order, start=1)}
    scored_at = _now_utc()
    records = []
    for idx, row in enumerate(rows):
        prod = production.get(_key(row), {})
        records.append(
            {
                **{field: row.get(field, "") for field in IDENTITY_FIELDS},
                "candidate_pool_source": row.get("candidate_pool_source") or "unknown",
                "lineup_source": _lineup_source(row),
                "production_rank": "" if not prod else str(_rank(prod, "learned_rank")),
                "production_probability": prod.get("learned_hit_probability", ""),
                "production_model_version": prod.get("model_version", ""),
                "production_model_trained_at": prod.get("model_trained_at", ""),
                "appearance_probability": f"{float(app_probability[idx]):.4f}",
                "hit_given_appearance_probability": f"{float(hit_probability[idx]):.4f}",
                "combined_probability_raw": f"{float(combined_raw[idx]):.4f}",
                "combined_probability_calibrated": f"{float(combined_calibrated[idx]):.4f}",
                "shadow_rank": str(shadow_ranks[idx]),
                "shadow_top5_rank": str(shadow_top5_ranks.get(idx, "")),
                "stuff_preference_rank": str(stuff_ranks.get(idx, "")),
                "stop_valve_count": str(len(_split_reasons(row.get("hard_pass_reasons")))),
                "model_version": model["model_version"],
                "model_trained_at": model["trained_at"],
                "scored_at": scored_at,
                "feature_snapshot_id": row.get("feature_snapshot_id", ""),
                "feature_source_hash": row.get("feature_source_hash", ""),
            }
        )
    _write_predictions(Path(out_csv), records)
    return sorted(records, key=lambda row: _rank(row, "shadow_rank"))


def _longest_miss_streak(labels: list[int]) -> int:
    longest = current = 0
    for label in labels:
        if label:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _policy_daily(
    predictions: list[dict],
    candidates: dict[tuple[str, str, str], dict],
    rank_field: str,
    *,
    top_n: int = 1,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        if _rank(prediction, "production_rank") <= 5:
            grouped[_date_text(prediction)].append(prediction)
    output = []
    for day in sorted(grouped):
        selected = sorted(grouped[day], key=lambda row: _rank(row, rank_field))[:top_n]
        labels = [decision_outcome(candidates.get(_key(row), {})) for row in selected]
        if not labels or any(label is None for label in labels):
            continue
        output.append(
            {
                "date": day,
                "hit": int(any(labels)),
                "both_hit": int(all(labels)) if top_n == 2 else None,
                "picks": [row.get("player") for row in selected],
            }
        )
    return output


def promotion_report(
    *,
    candidates_csv: str | Path = DEFAULT_OUTPUT_CSV,
    shadow_predictions_csv: str | Path = DEFAULT_SHADOW_PREDICTIONS_CSV,
    snapshots_csv: str | Path = DEFAULT_SNAPSHOTS_CSV,
    results_csv: str | Path = DEFAULT_RESULTS_CSV,
    out_json: str | Path = DEFAULT_PROMOTION_REPORT_JSON,
) -> dict:
    snapshots_path = Path(snapshots_csv)
    results_path = Path(results_csv)
    snapshots = load_rows(snapshots_path) if snapshots_path.exists() else []
    results = load_rows(results_path) if results_path.exists() else []
    eligible_snapshots = [
        row
        for row in snapshots
        if (_parse_date(_date_text(row)) or date.min) >= SHADOW_START_DATE
        and str(row.get("shadow_model_version") or "").strip()
    ]
    runs_by_date: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in eligible_snapshots:
        runs_by_date[_date_text(row)][str(row.get("run_id") or "")].append(row)
    predictions = []
    for day, runs in sorted(runs_by_date.items()):
        complete_runs = [rows for rows in runs.values() if rows]
        if not complete_runs:
            continue
        selected_run = min(complete_runs, key=lambda rows: str(rows[0].get("cutoff_at") or ""))
        predictions.extend(selected_run)

    if predictions:
        result_index = {
            (
                str(row.get("run_id") or ""),
                str(row.get("candidate_key") or ""),
                str(row.get("snapshot_hash") or ""),
            ): row
            for row in results
        }
        candidates = {
            _key(snapshot): result_index.get(
                (
                    str(snapshot.get("run_id") or ""),
                    str(snapshot.get("candidate_key") or ""),
                    str(snapshot.get("snapshot_hash") or ""),
                ),
                {},
            )
            for snapshot in predictions
        }
    else:
        candidates = {}
    prospective_top5: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        if _rank(prediction, "production_rank") <= 5:
            prospective_top5[_date_text(prediction)].append(prediction)
    fully_resolved_dates = {
        day
        for day, rows in prospective_top5.items()
        if len(rows) == 5 and all(decision_outcome(candidates.get(_key(row), {})) is not None for row in rows)
    }
    predictions = [row for row in predictions if _date_text(row) in fully_resolved_dates]
    policies = {
        "production_rank_one": _policy_daily(predictions, candidates, "production_rank"),
        "production_top_two": _policy_daily(predictions, candidates, "production_rank", top_n=2),
        "shadow_rank_one": _policy_daily(predictions, candidates, "shadow_top5_rank"),
        "stuff_preference_rank_one": _policy_daily(predictions, candidates, "stuff_preference_rank"),
        "shadow_top_two": _policy_daily(predictions, candidates, "shadow_top5_rank", top_n=2),
    }
    summaries = {}
    for name, rows in policies.items():
        labels = [row["hit"] for row in rows]
        summaries[name] = {
            "resolved_days": len(rows),
            "hits": sum(labels),
            "hit_rate": (sum(labels) / len(labels)) if labels else None,
            "longest_miss_streak": _longest_miss_streak(labels),
        }
        if name in {"production_top_two", "shadow_top_two"}:
            summaries[name]["both_hit_days"] = sum(row["both_hit"] for row in rows)
            summaries[name]["both_hit_rate"] = (
                sum(row["both_hit"] for row in rows) / len(rows) if rows else None
            )

    resolved_probabilities = []
    resolved_labels = []
    appearance_probabilities = []
    appearance_labels = []
    for prediction in predictions:
        result = candidates.get(_key(prediction), {})
        label = decision_outcome(result)
        probability = parse_float(prediction.get("combined_probability_calibrated"))
        if label is None or probability is None:
            continue
        resolved_labels.append(label)
        resolved_probabilities.append(probability)
        appearance = appearance_label(result)
        appearance_probability = parse_float(prediction.get("appearance_probability"))
        if appearance is not None and appearance_probability is not None:
            appearance_labels.append(appearance)
            appearance_probabilities.append(appearance_probability)
    y = np.array(resolved_labels, dtype=float)
    p = np.array(resolved_probabilities, dtype=float)
    appearance_y = np.array(appearance_labels, dtype=float)
    appearance_p = np.array(appearance_probabilities, dtype=float)
    resolved_days = summaries["shadow_rank_one"]["resolved_days"]
    payload = {
        "generated_at": _now_utc(),
        "shadow_start_date": SHADOW_START_DATE.isoformat(),
        "minimum_resolved_days": PROMOTION_MIN_RESOLVED_DAYS,
        "resolved_days": resolved_days,
        "remaining_days": max(0, PROMOTION_MIN_RESOLVED_DAYS - resolved_days),
        "eligible_for_promotion_review": resolved_days >= PROMOTION_MIN_RESOLVED_DAYS,
        "automatically_promoted": False,
        "status": "eligible_for_human_review" if resolved_days >= PROMOTION_MIN_RESOLVED_DAYS else "collecting",
        "metrics": summaries,
        "probability_metrics": {
            "resolved_predictions": len(y),
            "brier": _brier(y, p),
            "log_loss": _log_loss(y, p),
            "appearance_resolved_predictions": len(appearance_y),
            "appearance_brier": _brier(appearance_y, appearance_p),
        },
        "predeclared_metrics": [
            "rank_one_hit_rate",
            "top_two_any_hit_rate",
            "longest_miss_streak",
            "brier_score",
        "performance_by_season_and_month",
        ],
    }
    by_month: dict[str, list[int]] = defaultdict(list)
    for row in policies["shadow_rank_one"]:
        by_month[row["date"][:7]].append(row["hit"])
    payload["performance_by_month"] = [
        {"month": month, "hits": sum(values), "days": len(values), "hit_rate": sum(values) / len(values)}
        for month, values in sorted(by_month.items())
    ]
    by_season: dict[str, list[int]] = defaultdict(list)
    for row in policies["shadow_rank_one"]:
        by_season[row["date"][:4]].append(row["hit"])
    payload["performance_by_season"] = [
        {"season": season, "hits": sum(values), "days": len(values), "hit_rate": sum(values) / len(values)}
        for season, values in sorted(by_season.items())
    ]
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_shadow(
    *,
    candidates_csv: str | Path = DEFAULT_OUTPUT_CSV,
    production_predictions_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    model_out: str | Path = DEFAULT_SHADOW_MODEL_JSON,
    report_out: str | Path = DEFAULT_SHADOW_REPORT_JSON,
    predictions_out: str | Path = DEFAULT_SHADOW_PREDICTIONS_CSV,
    promotion_out: str | Path = DEFAULT_PROMOTION_REPORT_JSON,
    date_filter: str = "latest",
) -> dict:
    training = train_shadow_model(
        candidates_csv,
        production_predictions_csv=production_predictions_csv,
        model_out=model_out,
        report_out=report_out,
    )
    records = score_shadow_candidates(
        candidates_csv,
        production_predictions_csv=production_predictions_csv,
        model_path=model_out,
        out_csv=predictions_out,
        date_filter=date_filter,
    )
    promotion = promotion_report(
        candidates_csv=candidates_csv,
        shadow_predictions_csv=predictions_out,
        out_json=promotion_out,
    )
    return {"training": training, "records": records, "promotion": promotion}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the production-preserving Statbirt learned shadow model.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "score", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--candidates", default=str(DEFAULT_OUTPUT_CSV))
        sub.add_argument("--model", default=str(DEFAULT_SHADOW_MODEL_JSON))
        sub.add_argument("--report", default=str(DEFAULT_SHADOW_REPORT_JSON))
        sub.add_argument("--predictions", default=str(DEFAULT_SHADOW_PREDICTIONS_CSV))
        sub.add_argument("--production-predictions", default=str(DEFAULT_PREDICTIONS_CSV))
        sub.add_argument("--date", default="latest")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--candidates", default=str(DEFAULT_OUTPUT_CSV))
    evaluate.add_argument("--predictions", default=str(DEFAULT_SHADOW_PREDICTIONS_CSV))
    evaluate.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS_CSV))
    evaluate.add_argument("--results", default=str(DEFAULT_RESULTS_CSV))
    evaluate.add_argument("--out", default=str(DEFAULT_PROMOTION_REPORT_JSON))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        result = train_shadow_model(
            args.candidates,
            production_predictions_csv=args.production_predictions,
            model_out=args.model,
            report_out=args.report,
        )
        print(f"Trained {result['model']['model_version']} on {result['report']['hit_training_rows']} hit rows.")
    elif args.command == "score":
        records = score_shadow_candidates(
            args.candidates,
            production_predictions_csv=args.production_predictions,
            model_path=args.model,
            out_csv=args.predictions,
            date_filter=args.date,
        )
        print(f"Scored {len(records)} shadow candidates for {_date_text(records[0])}.")
    elif args.command == "run":
        result = run_shadow(
            candidates_csv=args.candidates,
            production_predictions_csv=args.production_predictions,
            model_out=args.model,
            report_out=args.report,
            predictions_out=args.predictions,
            date_filter=args.date,
        )
        print(
            f"Ran {result['training']['model']['model_version']}; "
            f"scored {len(result['records'])} candidates; "
            f"promotion gate {result['promotion']['resolved_days']}/{PROMOTION_MIN_RESOLVED_DAYS} days."
        )
    else:
        report = promotion_report(
            candidates_csv=args.candidates,
            shadow_predictions_csv=args.predictions,
            snapshots_csv=args.snapshots,
            results_csv=args.results,
            out_json=args.out,
        )
        print(
            f"Shadow promotion gate: {report['resolved_days']}/{report['minimum_resolved_days']} resolved days "
            f"({report['status']})."
        )


if __name__ == "__main__":
    main()
