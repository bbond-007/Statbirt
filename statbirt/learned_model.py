from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np

from .config import DATA_DIR, DEFAULT_OUTPUT_CSV
from .utils import normalize_name, parse_float

DEFAULT_MODEL_JSON = DATA_DIR / "models" / "hit_probability_model.json"
DEFAULT_REPORT_JSON = DATA_DIR / "models" / "hit_probability_report.json"
DEFAULT_PREDICTIONS_CSV = DATA_DIR / "model_predictions.csv"

IDENTITY_COLUMNS = [
    "date",
    "player",
    "player_id",
    "team",
    "opponent",
    "game_pk",
]

RESULT_COLUMNS = {
    "result_hit",
    "result_hits",
    "result_ab",
    "result_pa",
    "result_status",
    "result_updated_at",
    "notes",
}

NUMERIC_COLUMNS = [
    "lineup_slot",
    "expected_pa",
    "starts_last_5",
    "precip_probability",
    "forecast_temperature_f",
    "hitter_hipa_2500_pa",
    "hitter_pa_per_game_season",
    "hitter_ba_2500_ab",
    "hitter_hipa_500_pa",
    "hitter_hipa_75_ab",
    "hitter_ba_75_ab",
    "hitter_ba_25_ab",
    "hitter_ba_500_ab",
    "hitter_bb_rate_season",
    "hitter_bb_rate_500_pa",
    "hitter_whiff_rate_season",
    "hitter_whiff_rate_500_pa",
    "hitter_k_rate_season",
    "hitter_k_rate_500_pa",
    "hitter_split_ba_500_vs_lhp",
    "hitter_split_ba_500_vs_rhp",
    "hitter_split_ba_1500_vs_lhp",
    "hitter_split_ba_1500_vs_rhp",
    "pitcher_hpi_350",
    "pitcher_hpi_200",
    "pitcher_hpi_season",
    "pitcher_hits_last_18_ip",
    "pitcher_stuff_plus",
    "h2h_pa",
    "h2h_hit_rate",
    "h2h_whiff_rate",
    "h2h_k_rate",
    "h2h_exit_velocity",
    "h2h_xba",
    "pitcher_lr_opp_ba",
    "pitcher_lr_opp_ba_50",
    "pitcher_lr_opp_ba_200",
    "inferred_pitch_type_ba",
    "inferred_pitch_type_xba",
    "inferred_pitch_type_coverage",
    "bullpen_hpi",
    "bullpen_opp_ba",
    "sprint_speed",
    "park_hit_factor",
]

BOOLEAN_COLUMNS = [
    "confirmed_lineup",
    "road_game",
    "division_matchup",
    "doubleheader",
]

CATEGORICAL_COLUMNS = [
    "team",
    "opponent",
    "pitcher_hand",
    "batter_stand",
]

PREDICTION_FIELDS = [
    "date",
    "player",
    "player_id",
    "team",
    "opponent",
    "game_pk",
    "bob_score",
    "pickable",
    "learned_hit_probability",
    "learned_rank",
    "model_version",
    "model_trained_at",
    "result_hit",
    "result_hits",
    "result_ab",
    "result_pa",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_name(value)).strip("_")
    return slug or "unknown"


def _split_pipe(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _bool_value(value: str) -> float:
    text = str(value or "").strip().lower()
    return 1.0 if text in {"y", "yes", "true", "1"} else 0.0


def _label_value(row: dict[str, str]) -> int | None:
    value = str(row.get("result_hit") or "").strip()
    if value == "1":
        return 1
    if value == "0":
        return 0
    return None


def _date_value(row: dict[str, str]) -> str:
    return str(row.get("date") or "").strip()


def _date_parts(row: dict[str, str]) -> dict[str, float | None]:
    try:
        parsed = datetime.fromisoformat(_date_value(row)).date()
    except ValueError:
        return {"game_month": None, "game_day_of_week": None}
    return {
        "game_month": float(parsed.month),
        "game_day_of_week": float(parsed.weekday()),
    }


def load_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def labeled_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if _label_value(row) is not None]


def _category_value(row: dict[str, str], column: str) -> str:
    value = str(row.get(column) or "").strip()
    return value if value else "unknown"


def build_feature_spec(
    rows: list[dict[str, str]],
    *,
    min_category_count: int = 2,
    min_stop_reason_count: int = 2,
) -> dict:
    category_values: dict[str, list[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        counts = Counter(_category_value(row, column) for row in rows)
        values = sorted(value for value, count in counts.items() if count >= min_category_count)
        category_values[column] = values or ["unknown"]

    stop_counts = Counter(reason for row in rows for reason in _split_pipe(row.get("hard_pass_reasons", "")))
    stop_reasons = sorted(reason for reason, count in stop_counts.items() if count >= min_stop_reason_count)

    numeric_feature_names: list[str] = []
    medians: dict[str, float] = {}
    for column in NUMERIC_COLUMNS:
        name = f"num__{column}"
        numeric_feature_names.append(name)
        values = [parse_float(row.get(column)) for row in rows]
        observed = [value for value in values if value is not None]
        medians[name] = float(np.median(observed)) if observed else 0.0

    for name in ("num__game_month", "num__game_day_of_week", "num__stop_valve_count", "num__concern_count"):
        numeric_feature_names.append(name)
        medians[name] = 0.0

    feature_names: list[str] = []
    for name in numeric_feature_names:
        feature_names.append(name)
        if name.removeprefix("num__") in NUMERIC_COLUMNS:
            feature_names.append(f"missing__{name.removeprefix('num__')}")

    feature_names.extend(f"bool__{column}" for column in BOOLEAN_COLUMNS)

    for column, values in category_values.items():
        feature_names.extend(f"cat__{column}__{_slug(value)}" for value in values)

    feature_names.append("stop__has_any_stop_valve")
    feature_names.extend(f"stop__{_slug(reason)}" for reason in stop_reasons)

    return {
        "numeric_columns": NUMERIC_COLUMNS,
        "boolean_columns": BOOLEAN_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "category_values": category_values,
        "stop_reasons": stop_reasons,
        "numeric_medians": medians,
        "feature_names": feature_names,
    }


def _row_feature_values(row: dict[str, str], spec: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    medians = spec["numeric_medians"]

    for column in spec["numeric_columns"]:
        parsed = parse_float(row.get(column))
        name = f"num__{column}"
        values[name] = float(parsed) if parsed is not None else float(medians.get(name, 0.0))
        values[f"missing__{column}"] = 1.0 if parsed is None else 0.0

    for name, parsed in _date_parts(row).items():
        feature = f"num__{name}"
        values[feature] = float(parsed) if parsed is not None else float(medians.get(feature, 0.0))

    stop_reasons = set(_split_pipe(row.get("hard_pass_reasons", "")))
    concerns = _split_pipe(row.get("concerns", ""))
    values["num__stop_valve_count"] = float(len(stop_reasons))
    values["num__concern_count"] = float(len(concerns))

    for column in spec["boolean_columns"]:
        values[f"bool__{column}"] = _bool_value(row.get(column, ""))

    for column, allowed_values in spec["category_values"].items():
        current = _category_value(row, column)
        for allowed in allowed_values:
            values[f"cat__{column}__{_slug(allowed)}"] = 1.0 if current == allowed else 0.0

    values["stop__has_any_stop_valve"] = 1.0 if stop_reasons else 0.0
    for reason in spec["stop_reasons"]:
        values[f"stop__{_slug(reason)}"] = 1.0 if reason in stop_reasons else 0.0
    return values


def feature_matrix(rows: list[dict[str, str]], spec: dict) -> np.ndarray:
    feature_names = spec["feature_names"]
    matrix = np.zeros((len(rows), len(feature_names)), dtype=float)
    for idx, row in enumerate(rows):
        values = _row_feature_values(row, spec)
        matrix[idx] = [values.get(name, 0.0) for name in feature_names]
    return matrix


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (x - means) / stds, means, stds


def _standardize_apply(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (x - means) / stds


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _train_weights(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=float)
    intercept = 0.0
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    sample_weights = np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    weight_sum = float(sample_weights.sum())

    for step in range(iterations):
        probabilities = _sigmoid(x @ weights + intercept)
        error = (probabilities - y) * sample_weights
        gradient = (x.T @ error) / weight_sum + l2 * weights
        intercept_gradient = float(error.sum() / weight_sum)
        rate = learning_rate / math.sqrt(1.0 + step / 500.0)
        weights -= rate * gradient
        intercept -= rate * intercept_gradient
    return weights, intercept


def _predict_probability(model: dict, rows: list[dict[str, str]]) -> np.ndarray:
    spec = model["feature_spec"]
    x = feature_matrix(rows, spec)
    means = np.array(model["standardization"]["means"], dtype=float)
    stds = np.array(model["standardization"]["stds"], dtype=float)
    x_std = _standardize_apply(x, means, stds)
    weights = np.array(model["weights"], dtype=float)
    intercept = float(model["intercept"])
    return _sigmoid(x_std @ weights + intercept)


def _log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())


def _brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - y) ** 2))


def _accuracy(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities >= 0.5) == y))


def _roc_auc(y: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = probabilities[y == 1]
    negatives = probabilities[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    wins = 0.0
    for value in positives:
        wins += float(np.sum(value > negatives))
        wins += 0.5 * float(np.sum(value == negatives))
    return wins / float(len(positives) * len(negatives))


def _top_n_hit_rate(rows: list[dict[str, str]], y: np.ndarray, probabilities: np.ndarray, *, n: int) -> float | None:
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, label, probability in zip(rows, y, probabilities, strict=True):
        grouped[_date_value(row)].append((float(probability), int(label)))
    labels = []
    for values in grouped.values():
        values.sort(reverse=True)
        labels.extend(label for _, label in values[:n])
    if not labels:
        return None
    return float(np.mean(labels))


def _metrics(rows: list[dict[str, str]], y: np.ndarray, probabilities: np.ndarray) -> dict:
    output = {
        "rows": len(rows),
        "hit_rate": float(np.mean(y)) if len(y) else None,
        "accuracy_0_50": _accuracy(y, probabilities) if len(y) else None,
        "log_loss": _log_loss(y, probabilities) if len(y) else None,
        "brier": _brier(y, probabilities) if len(y) else None,
        "roc_auc": _roc_auc(y, probabilities) if len(y) else None,
        "top_10_hit_rate": _top_n_hit_rate(rows, y, probabilities, n=10) if len(y) else None,
    }
    return output


def _date_split(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dates = sorted({_date_value(row) for row in rows if _date_value(row)})
    if len(dates) < 4:
        return rows, []
    cutoff = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    train_dates = set(dates[:cutoff])
    return [row for row in rows if _date_value(row) in train_dates], [row for row in rows if _date_value(row) not in train_dates]


def _fit_model(
    rows: list[dict[str, str]],
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> dict:
    spec = build_feature_spec(rows)
    x = feature_matrix(rows, spec)
    y = np.array([_label_value(row) for row in rows], dtype=float)
    if len(set(y.astype(int).tolist())) < 2:
        raise ValueError("Training data must include both hit and no-hit labels.")
    x_std, means, stds = _standardize_fit(x)
    weights, intercept = _train_weights(x_std, y, iterations=iterations, learning_rate=learning_rate, l2=l2)
    probabilities = _sigmoid(x_std @ weights + intercept)
    return {
        "feature_spec": spec,
        "standardization": {"means": means.tolist(), "stds": stds.tolist()},
        "weights": weights.tolist(),
        "intercept": intercept,
        "training_metrics": _metrics(rows, y, probabilities),
    }


def _correlations(rows: list[dict[str, str]]) -> list[dict]:
    y = np.array([_label_value(row) for row in rows], dtype=float)
    output = []
    for column in NUMERIC_COLUMNS:
        pairs = [(parse_float(row.get(column)), label) for row, label in zip(rows, y, strict=True)]
        pairs = [(value, label) for value, label in pairs if value is not None]
        if len(pairs) < 10:
            continue
        x = np.array([value for value, _ in pairs], dtype=float)
        labels = np.array([label for _, label in pairs], dtype=float)
        if float(np.std(x)) < 1e-8 or float(np.std(labels)) < 1e-8:
            continue
        corr = float(np.corrcoef(x, labels)[0, 1])
        output.append({"feature": column, "correlation": corr, "rows": len(pairs)})
    return sorted(output, key=lambda row: abs(row["correlation"]), reverse=True)


def _stop_reason_summary(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        label = _label_value(row)
        if label is None:
            continue
        reasons = _split_pipe(row.get("hard_pass_reasons", ""))
        if not reasons:
            grouped["No stop valve"].append(label)
        for reason in reasons:
            grouped[reason].append(label)
    output = []
    for reason, labels in grouped.items():
        if len(labels) < 5:
            continue
        output.append({"reason": reason, "rows": len(labels), "hit_rate": float(np.mean(labels))})
    return sorted(output, key=lambda row: row["hit_rate"])


def _weight_summary(model: dict, *, limit: int = 20) -> dict:
    names = model["feature_spec"]["feature_names"]
    weights = np.array(model["weights"], dtype=float)
    paired = [{"feature": name, "weight": float(weight)} for name, weight in zip(names, weights, strict=True)]
    return {
        "positive": sorted(paired, key=lambda row: row["weight"], reverse=True)[:limit],
        "negative": sorted(paired, key=lambda row: row["weight"])[:limit],
    }


def train_model(
    candidates_csv: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    model_out: str | Path = DEFAULT_MODEL_JSON,
    report_out: str | Path = DEFAULT_REPORT_JSON,
    min_rows: int = 200,
    iterations: int = 2500,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> dict:
    all_rows = load_rows(candidates_csv)
    rows = labeled_rows(all_rows)
    if len(rows) < min_rows:
        raise ValueError(f"Need at least {min_rows} labeled rows to train; found {len(rows)}.")

    train_rows, validation_rows = _date_split(rows)
    validation_metrics = None
    if validation_rows and len({_label_value(row) for row in train_rows}) == 2:
        split_model = _fit_model(train_rows, iterations=iterations, learning_rate=learning_rate, l2=l2)
        y_val = np.array([_label_value(row) for row in validation_rows], dtype=float)
        validation_metrics = _metrics(validation_rows, y_val, _predict_probability(split_model, validation_rows))

    trained_at = _now_utc()
    model = _fit_model(rows, iterations=iterations, learning_rate=learning_rate, l2=l2)
    model.update(
        {
            "model_type": "standardized_l2_logistic_regression",
            "model_version": f"learned-logistic-v1-{trained_at.replace(':', '').replace('-', '')}",
            "trained_at": trained_at,
            "candidates_csv": str(Path(candidates_csv).resolve()),
            "training_rows": len(rows),
            "training_date_min": min(_date_value(row) for row in rows if _date_value(row)),
            "training_date_max": max(_date_value(row) for row in rows if _date_value(row)),
            "parameters": {
                "iterations": iterations,
                "learning_rate": learning_rate,
                "l2": l2,
                "min_rows": min_rows,
            },
        }
    )

    report = {
        "model_version": model["model_version"],
        "trained_at": trained_at,
        "available_candidate_rows": len(all_rows),
        "labeled_training_rows": len(rows),
        "training_date_min": model["training_date_min"],
        "training_date_max": model["training_date_max"],
        "training_metrics": model["training_metrics"],
        "validation_metrics": validation_metrics,
        "feature_weights": _weight_summary(model),
        "numeric_correlations": _correlations(rows)[:30],
        "stop_reason_hit_rates": _stop_reason_summary(rows),
    }

    model_path = Path(model_out)
    report_path = Path(report_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"model": model, "report": report, "model_path": model_path, "report_path": report_path}


def load_model(path: str | Path = DEFAULT_MODEL_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _filter_rows_for_date(rows: list[dict[str, str]], date_filter: str | None) -> list[dict[str, str]]:
    if not date_filter or date_filter == "all":
        return rows
    dates = sorted({_date_value(row) for row in rows if _date_value(row)})
    if not dates:
        return []
    target = dates[-1] if date_filter == "latest" else date_filter
    return [row for row in rows if _date_value(row) == target]


def _prediction_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("date") or ""),
        str(row.get("player_id") or ""),
        str(row.get("game_pk") or ""),
    )


def _write_predictions(path: Path, records: list[dict[str, str]], *, replace_output: bool) -> None:
    existing = [] if replace_output else load_rows(path)
    new_keys = {_prediction_key(row) for row in records}
    merged = [row for row in existing if _prediction_key(row) not in new_keys]
    merged.extend(records)

    def sort_key(row: dict[str, str]):
        try:
            rank = int(row.get("learned_rank") or 999999)
        except ValueError:
            rank = 999999
        return (row.get("date") or "", row.get("model_version") or "", -rank)

    merged.sort(key=sort_key, reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in merged:
            writer.writerow({field: row.get(field, "") for field in PREDICTION_FIELDS})


def score_candidates(
    candidates_csv: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    model_path: str | Path = DEFAULT_MODEL_JSON,
    out_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    date_filter: str | None = "latest",
    replace_output: bool = False,
) -> list[dict[str, str]]:
    rows = _filter_rows_for_date(load_rows(candidates_csv), date_filter)
    if not rows:
        raise ValueError("No candidate rows matched the requested date filter.")

    model = load_model(model_path)
    probabilities = _predict_probability(model, rows)
    grouped_indexes: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped_indexes[_date_value(row)].append(idx)

    ranks = [0] * len(rows)
    for indexes in grouped_indexes.values():
        indexes.sort(key=lambda idx: probabilities[idx], reverse=True)
        for rank, idx in enumerate(indexes, start=1):
            ranks[idx] = rank

    records = []
    for row, probability, rank in zip(rows, probabilities, ranks, strict=True):
        records.append(
            {
                "date": row.get("date", ""),
                "player": row.get("player", ""),
                "player_id": row.get("player_id", ""),
                "team": row.get("team", ""),
                "opponent": row.get("opponent", ""),
                "game_pk": row.get("game_pk", ""),
                "bob_score": row.get("score", ""),
                "pickable": row.get("pickable", ""),
                "learned_hit_probability": f"{float(probability):.4f}",
                "learned_rank": str(rank),
                "model_version": model["model_version"],
                "model_trained_at": model["trained_at"],
                "result_hit": row.get("result_hit", ""),
                "result_hits": row.get("result_hits", ""),
                "result_ab": row.get("result_ab", ""),
                "result_pa": row.get("result_pa", ""),
            }
        )

    _write_predictions(Path(out_csv), records, replace_output=replace_output)
    return sorted(records, key=lambda row: int(row["learned_rank"]))


def audit_candidates(candidates_csv: str | Path = DEFAULT_OUTPUT_CSV) -> dict:
    rows = load_rows(candidates_csv)
    labels = Counter(row.get("result_hit", "") for row in rows)
    by_date: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_date[_date_value(row)][row.get("result_hit", "")] += 1
    return {
        "rows": len(rows),
        "labeled_rows": sum(labels[value] for value in ("0", "1")),
        "labels": dict(labels),
        "date_min": min((_date_value(row) for row in rows if _date_value(row)), default=""),
        "date_max": max((_date_value(row) for row in rows if _date_value(row)), default=""),
        "by_date": {date_key: dict(counts) for date_key, counts in sorted(by_date.items())},
    }


def _print_training_summary(result: dict) -> None:
    report = result["report"]
    print(f"Trained {report['model_version']}")
    print(f"Rows: {report['labeled_training_rows']} labeled candidate rows")
    print(f"Dates: {report['training_date_min']} to {report['training_date_max']}")
    validation = report.get("validation_metrics") or {}
    if validation:
        auc = validation.get("roc_auc")
        top10 = validation.get("top_10_hit_rate")
        auc_text = "N/A" if auc is None else f"{auc:.3f}"
        top10_text = "N/A" if top10 is None else f"{top10:.3f}"
        print(
            "Validation: "
            f"AUC {auc_text} | "
            f"Brier {validation.get('brier'):.3f} | "
            f"Top 10 hit rate {top10_text}"
        )
    print(f"Model: {result['model_path']}")
    print(f"Report: {result['report_path']}")


def _print_predictions(records: list[dict[str, str]], *, top: int) -> None:
    print(f"Top {min(top, len(records))} learned-model picks")
    print("=" * 92)
    print(f"{'Rank':>4} {'Player':<24} {'Tm':<3} {'Opp':<3} {'Prob':>7} {'Bob':>7} {'Pick':>4}")
    print("-" * 92)
    for row in records[:top]:
        print(
            f"{row['learned_rank']:>4} "
            f"{row['player']:<24} "
            f"{row['team']:<3} "
            f"{row['opponent']:<3} "
            f"{float(row['learned_hit_probability']):>7.3f} "
            f"{row['bob_score']:>7} "
            f"{row['pickable']:>4}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train and run Statbirt's learned hit-probability model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train from labeled candidate rows.")
    train.add_argument("--candidates", default=str(DEFAULT_OUTPUT_CSV))
    train.add_argument("--model-out", default=str(DEFAULT_MODEL_JSON))
    train.add_argument("--report-out", default=str(DEFAULT_REPORT_JSON))
    train.add_argument("--min-rows", type=int, default=200)
    train.add_argument("--iterations", type=int, default=2500)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.add_argument("--l2", type=float, default=0.01)

    score = subparsers.add_parser("score", help="Score candidate rows with a trained model.")
    score.add_argument("--candidates", default=str(DEFAULT_OUTPUT_CSV))
    score.add_argument("--model", default=str(DEFAULT_MODEL_JSON))
    score.add_argument("--out", default=str(DEFAULT_PREDICTIONS_CSV))
    score.add_argument("--date", default="latest", help="YYYY-MM-DD, latest, or all.")
    score.add_argument("--replace-output", action="store_true")
    score.add_argument("--top", type=int, default=25)

    run = subparsers.add_parser("run", help="Train, then score candidate rows.")
    run.add_argument("--candidates", default=str(DEFAULT_OUTPUT_CSV))
    run.add_argument("--model-out", default=str(DEFAULT_MODEL_JSON))
    run.add_argument("--report-out", default=str(DEFAULT_REPORT_JSON))
    run.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS_CSV))
    run.add_argument("--date", default="latest", help="YYYY-MM-DD, latest, or all.")
    run.add_argument("--min-rows", type=int, default=200)
    run.add_argument("--iterations", type=int, default=2500)
    run.add_argument("--learning-rate", type=float, default=0.08)
    run.add_argument("--l2", type=float, default=0.01)
    run.add_argument("--top", type=int, default=25)

    audit = subparsers.add_parser("audit", help="Summarize candidate/result coverage.")
    audit.add_argument("--candidates", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        result = train_model(
            args.candidates,
            model_out=args.model_out,
            report_out=args.report_out,
            min_rows=args.min_rows,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        _print_training_summary(result)
    elif args.command == "score":
        records = score_candidates(
            args.candidates,
            model_path=args.model,
            out_csv=args.out,
            date_filter=args.date,
            replace_output=args.replace_output,
        )
        _print_predictions(records, top=args.top)
        print(f"\nWrote predictions to {Path(args.out).resolve()}")
    elif args.command == "run":
        result = train_model(
            args.candidates,
            model_out=args.model_out,
            report_out=args.report_out,
            min_rows=args.min_rows,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        _print_training_summary(result)
        print()
        records = score_candidates(
            args.candidates,
            model_path=args.model_out,
            out_csv=args.predictions_out,
            date_filter=args.date,
        )
        _print_predictions(records, top=args.top)
        print(f"\nWrote predictions to {Path(args.predictions_out).resolve()}")
    elif args.command == "audit":
        print(json.dumps(audit_candidates(args.candidates), indent=2))


if __name__ == "__main__":
    main()
