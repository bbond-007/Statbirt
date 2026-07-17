from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = Path("data/statbirt_candidates.csv")
DEFAULT_OUT_DIR = Path("logs/model_backtests")

BASE_NUMERIC_COLUMNS = [
    "lineup_slot",
    "expected_pa",
    "starts_last_5",
    "hitter_last_5_games_played",
    "hitter_last_5_games_hits",
    "hitter_last_5_games_ab",
    "hitter_last_5_games_ba",
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

COMPONENT_COLUMNS = [
    "component_hitter.hipa_2500_pa",
    "component_hitter.pa_per_game_season",
    "component_hitter.hipa_500_pa",
    "component_hitter.hipa_75_ab",
    "component_starting_pitcher.hpi_350",
    "component_starting_pitcher.hpi_season",
    "component_starting_pitcher.stuff_plus",
    "component_h2h.direct",
    "component_h2h.pitcher_lr_opp_ba",
    "component_h2h.inferred_pitch_type",
    "component_bullpen.hpi_season",
    "component_other.road_game",
    "component_other.division_matchup",
    "component_other.sprint_speed",
    "component_other.park_hit_factor",
    "component_other.lineup_opportunity",
]

OPPORTUNITY_CONTACT_COLUMNS = [
    "score",
    "expected_pa",
    "lineup_slot",
    "starts_last_5",
    "hitter_last_5_games_played",
    "hitter_last_5_games_hits",
    "hitter_last_5_games_ab",
    "hitter_last_5_games_ba",
    "hitter_hipa_2500_pa",
    "hitter_pa_per_game_season",
    "hitter_ba_2500_ab",
    "hitter_hipa_500_pa",
    "hitter_ba_500_ab",
    "hitter_hipa_75_ab",
    "hitter_ba_75_ab",
    "hitter_ba_25_ab",
    "hitter_whiff_rate_season",
    "hitter_whiff_rate_500_pa",
    "hitter_k_rate_season",
    "hitter_k_rate_500_pa",
    "pitcher_hpi_350",
    "pitcher_hpi_200",
    "pitcher_hpi_season",
    "pitcher_hits_last_18_ip",
    "pitcher_stuff_plus",
    "pitcher_lr_opp_ba",
    "pitcher_lr_opp_ba_50",
    "pitcher_lr_opp_ba_200",
    "h2h_pa",
    "h2h_hit_rate",
    "h2h_xba",
    "inferred_pitch_type_ba",
    "inferred_pitch_type_xba",
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
    "pickable",
]

CATEGORICAL_COLUMNS = [
    "team",
    "opponent",
    "pitcher_hand",
    "batter_stand",
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_") or "unknown"


def split_pipe(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def bool_value(value: object) -> float:
    return 1.0 if str(value or "").strip().lower() in {"y", "yes", "true", "1"} else 0.0


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def max_success_streak(labels: list[int]) -> int:
    current = 0
    best = 0
    for label in labels:
        if label == 1:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def ending_success_streak(labels: list[int]) -> int:
    current = 0
    for label in reversed(labels):
        if label == 1:
            current += 1
        else:
            break
    return current


def selected_numeric_columns(columns: list[str], available: set[str]) -> list[str]:
    return [column for column in columns if column in available]


def build_stop_reason_frame(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for value in df.get("hard_pass_reasons", pd.Series([""] * len(df))):
        for reason in split_pipe(value):
            counts[reason] = counts.get(reason, 0) + 1
    reasons = [reason for reason, count in sorted(counts.items()) if count >= min_count]
    data = {}
    source = df.get("hard_pass_reasons", pd.Series([""] * len(df))).fillna("")
    for reason in reasons:
        data[f"stop_reason__{slug(reason)}"] = source.map(lambda value, reason=reason: 1.0 if reason in split_pipe(value) else 0.0)
    return pd.DataFrame(data, index=df.index)


def one_hot_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frames = []
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].fillna("").astype(str).str.strip().replace("", "unknown")
        frames.append(pd.get_dummies(values, prefix=f"cat__{column}", dtype=float))
    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1)


def prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["label"] = df["result_hit"].map({"1": 1.0, "0": 0.0})
    for column in set(BASE_NUMERIC_COLUMNS + COMPONENT_COLUMNS + OPPORTUNITY_CONTACT_COLUMNS + ["score"]):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in BOOLEAN_COLUMNS:
        if column in df.columns:
            df[f"bool__{column}"] = df[column].map(bool_value)
    df["stop_valve_count"] = df.get("hard_pass_reasons", "").map(lambda value: len(split_pipe(value)))
    df["concern_count"] = df.get("concerns", "").map(lambda value: len(split_pipe(value)))
    df["date"] = df["date"].astype(str)
    return df


@dataclass
class MatrixSpec:
    name: str
    numeric_columns: list[str]
    boolean_columns: list[str] | None = None
    include_missing_indicators: bool = True
    include_booleans: bool = True
    include_categories: bool = False
    include_stop_counts: bool = False
    include_stop_reasons: bool = False
    balanced_loss: bool = False
    iterations: int = 260
    learning_rate: float = 0.06
    l2: float = 0.02


class MatrixBuilder:
    def __init__(self, df: pd.DataFrame, spec: MatrixSpec):
        self.df = df
        self.spec = spec
        frames = []
        self.numeric_columns = selected_numeric_columns(spec.numeric_columns, set(df.columns))
        if self.numeric_columns:
            frames.append(df[self.numeric_columns].astype(float))
            if spec.include_missing_indicators:
                frames.append(
                    df[self.numeric_columns]
                    .isna()
                    .astype(float)
                    .rename(columns={column: f"missing__{column}" for column in self.numeric_columns})
                )
        if spec.include_stop_counts:
            frames.append(df[["stop_valve_count", "concern_count"]].astype(float))
        if spec.include_booleans:
            source_boolean_columns = spec.boolean_columns if spec.boolean_columns is not None else BOOLEAN_COLUMNS
            bool_columns = [f"bool__{column}" for column in source_boolean_columns if f"bool__{column}" in df.columns]
            if bool_columns:
                frames.append(df[bool_columns].astype(float))
        if spec.include_categories:
            frames.append(one_hot_frame(df, CATEGORICAL_COLUMNS))
        if spec.include_stop_reasons:
            frames.append(build_stop_reason_frame(df))
        self.raw = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
        self.raw = self.raw.loc[:, ~self.raw.columns.duplicated()]
        self.columns = list(self.raw.columns)

    def train_test_arrays(self, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        train = self.raw.iloc[train_idx].astype(float)
        test = self.raw.iloc[test_idx].astype(float)
        medians = train.median(axis=0, skipna=True).fillna(0.0)
        train = train.fillna(medians)
        test = test.fillna(medians)
        means = train.mean(axis=0)
        stds = train.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return ((train - means) / stds).to_numpy(float), ((test - means) / stds).to_numpy(float)


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
    balanced_loss: bool,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=float)
    prevalence = min(max(float(y.mean()), 1e-4), 1 - 1e-4)
    intercept = math.log(prevalence / (1 - prevalence))
    if balanced_loss:
        positives = max(float(y.sum()), 1.0)
        negatives = max(float(len(y) - y.sum()), 1.0)
        sample_weights = np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    else:
        sample_weights = np.ones(len(y), dtype=float)
    weight_sum = float(sample_weights.sum())
    for step in range(iterations):
        probability = sigmoid(x @ weights + intercept)
        error = (probability - y) * sample_weights
        gradient = (x.T @ error) / weight_sum + l2 * weights
        intercept_gradient = float(error.sum() / weight_sum)
        rate = learning_rate / math.sqrt(1.0 + step / 350.0)
        weights -= rate * gradient
        intercept -= rate * intercept_gradient
    return weights, intercept


def ranker_scores(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    def z(column: str, default: float = 0.0) -> np.ndarray:
        train_values = pd.to_numeric(train.get(column, pd.Series(index=train.index, dtype=float)), errors="coerce")
        test_values = pd.to_numeric(test.get(column, pd.Series(index=test.index, dtype=float)), errors="coerce")
        median = float(train_values.median(skipna=True)) if train_values.notna().any() else default
        filled_train = train_values.fillna(median)
        mean = float(filled_train.mean())
        std = float(filled_train.std(ddof=0))
        if not math.isfinite(std) or std < 1e-8:
            std = 1.0
        return ((test_values.fillna(median) - mean) / std).to_numpy(float)

    score = np.zeros(len(test), dtype=float)
    score += 0.26 * z("score")
    score += 0.18 * z("expected_pa")
    score -= 0.12 * z("lineup_slot")
    score += 0.10 * z("starts_last_5")
    score += 0.12 * z("hitter_pa_per_game_season")
    score += 0.11 * z("hitter_hipa_2500_pa")
    score += 0.09 * z("hitter_hipa_500_pa")
    score += 0.07 * z("hitter_ba_500_ab")
    score += 0.05 * z("hitter_last_5_games_ba")
    score += 0.10 * z("pitcher_hpi_350")
    score += 0.08 * z("pitcher_hpi_season")
    score += 0.08 * z("pitcher_lr_opp_ba")
    score += 0.05 * z("bullpen_opp_ba")
    score += 0.05 * z("h2h_xba")
    score += 0.04 * z("inferred_pitch_type_ba")
    score -= 0.06 * z("pitcher_stuff_plus")
    score -= 0.08 * z("hitter_k_rate_500_pa")
    score -= 0.05 * z("hitter_whiff_rate_500_pa")
    score -= 0.05 * z("hitter_bb_rate_500_pa")
    score -= 0.04 * z("stop_valve_count")
    score -= 0.02 * z("concern_count")
    if "bool__confirmed_lineup" in df.columns:
        score += 0.05 * test["bool__confirmed_lineup"].to_numpy(float)
    return score


def indicator_ranker_scores(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    def z(column: str, default: float = 0.0) -> np.ndarray:
        train_values = pd.to_numeric(train.get(column, pd.Series(index=train.index, dtype=float)), errors="coerce")
        test_values = pd.to_numeric(test.get(column, pd.Series(index=test.index, dtype=float)), errors="coerce")
        median = float(train_values.median(skipna=True)) if train_values.notna().any() else default
        filled_train = train_values.fillna(median)
        mean = float(filled_train.mean())
        std = float(filled_train.std(ddof=0))
        if not math.isfinite(std) or std < 1e-8:
            std = 1.0
        return ((test_values.fillna(median) - mean) / std).to_numpy(float)

    score = np.zeros(len(test), dtype=float)
    score += 0.23 * z("hitter_pa_per_game_season")
    score += 0.18 * z("expected_pa")
    score -= 0.18 * z("lineup_slot")
    score += 0.17 * z("score")
    score += 0.14 * z("hitter_last_5_games_ab")
    score += 0.12 * z("hitter_ba_500_ab")
    score += 0.11 * z("hitter_ba_2500_ab")
    score += 0.09 * z("hitter_hipa_500_pa")
    score += 0.08 * z("hitter_hipa_2500_pa")
    score += 0.07 * z("hitter_last_5_games_hits")
    score += 0.06 * z("hitter_split_ba_500_vs_rhp")
    score += 0.06 * z("hitter_split_ba_1500_vs_rhp")
    score += 0.06 * z("pitcher_hpi_season")
    score += 0.05 * z("inferred_pitch_type_ba")
    score += 0.04 * z("pitcher_lr_opp_ba")
    score -= 0.06 * z("hitter_k_rate_500_pa")
    score -= 0.04 * z("hitter_whiff_rate_500_pa")
    score -= 0.03 * z("pitcher_stuff_plus")
    return score


def bob_score_scores(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    train = pd.to_numeric(df.iloc[train_idx].get("score"), errors="coerce")
    test = pd.to_numeric(df.iloc[test_idx].get("score"), errors="coerce")
    median = float(train.median(skipna=True)) if train.notna().any() else 0.0
    filled_train = train.fillna(median)
    mean = float(filled_train.mean())
    std = float(filled_train.std(ddof=0))
    if not math.isfinite(std) or std < 1e-8:
        std = 1.0
    return ((test.fillna(median) - mean) / std).to_numpy(float)


@dataclass
class Experiment:
    name: str
    description: str
    matrix_spec: MatrixSpec | None = None
    score_func: Callable[[pd.DataFrame, np.ndarray, np.ndarray], np.ndarray] | None = None


def evaluate_experiment(df: pd.DataFrame, experiment: Experiment, *, min_train_dates: int) -> tuple[dict, list[dict]]:
    labeled = df[df["label"].notna()].copy()
    dates = sorted(labeled["date"].unique())
    builder = MatrixBuilder(df, experiment.matrix_spec) if experiment.matrix_spec is not None else None
    daily: list[dict] = []
    labels: list[int] = []
    raw_ungraded_count = 0

    for date_value in dates[min_train_dates:]:
        train_dates = [date for date in dates if date < date_value]
        if len(train_dates) < min_train_dates:
            continue
        train_idx = df.index[(df["date"].isin(train_dates)) & df["label"].notna()].to_numpy()
        test_idx = df.index[df["date"].eq(date_value)].to_numpy()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        if builder is not None and experiment.matrix_spec is not None:
            x_train, x_test = builder.train_test_arrays(train_idx, test_idx)
            y_train = df.loc[train_idx, "label"].to_numpy(float)
            weights, intercept = fit_logistic(
                x_train,
                y_train,
                iterations=experiment.matrix_spec.iterations,
                learning_rate=experiment.matrix_spec.learning_rate,
                l2=experiment.matrix_spec.l2,
                balanced_loss=experiment.matrix_spec.balanced_loss,
            )
            scores = sigmoid(x_test @ weights + intercept)
        elif experiment.score_func is not None:
            scores = experiment.score_func(df, train_idx, test_idx)
        else:
            raise ValueError(f"Experiment {experiment.name} has no scorer.")

        test_rows = df.loc[test_idx].copy()
        test_rows["_score"] = scores
        test_rows = test_rows.sort_values("_score", ascending=False)

        raw_top = test_rows.iloc[0]
        raw_label = raw_top.get("label")
        if pd.isna(raw_label):
            raw_ungraded_count += 1
            continue

        pick = raw_top
        label = int(raw_label)
        labels.append(label)
        daily.append(
            {
                "experiment": experiment.name,
                "date": date_value,
                "player": pick.get("player", ""),
                "team": pick.get("team", ""),
                "opponent": pick.get("opponent", ""),
                "score": float(pick["_score"]),
                "bob_score": "" if pd.isna(pick.get("score")) else float(pick.get("score")),
                "pickable": pick.get("pickable", ""),
                "result_hit": label,
                "result_status": pick.get("result_status", ""),
                "raw_top_player": raw_top.get("player", ""),
                "raw_top_result_hit": "" if pd.isna(raw_label) else int(raw_label),
                "raw_top_result_status": raw_top.get("result_status", ""),
                "raw_top_score": float(raw_top["_score"]),
            }
        )

    wins = sum(labels)
    losses = len(labels) - wins
    summary = {
        "name": experiment.name,
        "description": experiment.description,
        "evaluated_dates": len(labels),
        "wins": wins,
        "losses": losses,
        "daily_success_rate": wins / len(labels) if labels else None,
        "max_success_streak": max_success_streak(labels),
        "ending_success_streak": ending_success_streak(labels),
        "raw_ungraded_top_count": raw_ungraded_count,
    }
    return summary, daily


def feature_univariate_summary(df: pd.DataFrame) -> list[dict]:
    labeled = df[df["label"].notna()].copy()
    output = []
    candidate_columns = selected_numeric_columns(["score", *BASE_NUMERIC_COLUMNS, *COMPONENT_COLUMNS], set(df.columns))
    for column in candidate_columns:
        values = pd.to_numeric(labeled[column], errors="coerce")
        usable = labeled[values.notna()].copy()
        if len(usable) < 300 or usable["label"].nunique() < 2:
            continue
        usable["_feature"] = pd.to_numeric(usable[column], errors="coerce")
        corr = float(usable["_feature"].corr(usable["label"])) if usable["_feature"].std(ddof=0) > 0 else 0.0
        try:
            bins = pd.qcut(usable["_feature"], q=5, duplicates="drop")
            grouped = usable.groupby(bins, observed=True)["label"].agg(["count", "mean"])
            low = float(grouped["mean"].iloc[0])
            high = float(grouped["mean"].iloc[-1])
        except Exception:
            low = None
            high = None
        output.append(
            {
                "feature": column,
                "rows": int(len(usable)),
                "correlation": corr,
                "bottom_bin_hit_rate": low,
                "top_bin_hit_rate": high,
                "top_minus_bottom": None if low is None or high is None else high - low,
            }
        )
    output.sort(key=lambda row: abs(row["top_minus_bottom"] or 0.0), reverse=True)
    return output


def stop_reason_summary(df: pd.DataFrame) -> list[dict]:
    labeled = df[df["label"].notna()].copy()
    grouped: dict[str, list[float]] = {"No stop valve": []}
    for _, row in labeled.iterrows():
        reasons = split_pipe(row.get("hard_pass_reasons", ""))
        if not reasons:
            grouped["No stop valve"].append(float(row["label"]))
        for reason in reasons:
            grouped.setdefault(reason, []).append(float(row["label"]))
    output = []
    for reason, labels in grouped.items():
        if len(labels) < 25:
            continue
        output.append({"reason": reason, "rows": len(labels), "hit_rate": sum(labels) / len(labels)})
    return sorted(output, key=lambda row: row["hit_rate"], reverse=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward Statbirt learned-model experiments.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-train-dates", type=int, default=30)
    parser.add_argument(
        "--experiments",
        default="",
        help="Optional comma-separated experiment names; defaults to the full suite.",
    )
    args = parser.parse_args()

    df = prepare_dataframe(Path(args.candidates))
    available = set(df.columns)
    experiments = [
        Experiment(
            name="iteration_0_bob_score",
            description="Control: rank by the existing primary-model score.",
            score_func=bob_score_scores,
        ),
        Experiment(
            name="iteration_1_current_like",
            description="Current-like logistic: broad numeric set, categories, stop counts/reasons, balanced loss.",
            matrix_spec=MatrixSpec(
                name="current_like",
                numeric_columns=selected_numeric_columns([*BASE_NUMERIC_COLUMNS], available),
                include_missing_indicators=True,
                include_booleans=True,
                include_categories=True,
                include_stop_counts=True,
                include_stop_reasons=True,
                balanced_loss=True,
                iterations=260,
                learning_rate=0.06,
                l2=0.02,
            ),
        ),
        Experiment(
            name="iteration_2_destopped_with_bob",
            description="Stop-neutral logistic: Bob score plus baseball numerics/booleans, no stop reason or team/opponent one-hots.",
            matrix_spec=MatrixSpec(
                name="destopped_with_bob",
                numeric_columns=selected_numeric_columns(["score", *BASE_NUMERIC_COLUMNS], available),
                boolean_columns=["confirmed_lineup", "road_game", "division_matchup", "doubleheader", "pickable"],
                include_missing_indicators=True,
                include_booleans=True,
                include_categories=False,
                include_stop_counts=False,
                include_stop_reasons=False,
                balanced_loss=False,
                iterations=300,
                learning_rate=0.06,
                l2=0.02,
            ),
        ),
        Experiment(
            name="iteration_3_opportunity_contact",
            description="Trimmed logistic: opportunity, contact floor, pitcher vulnerability, and Bob score; includes non-stop booleans only.",
            matrix_spec=MatrixSpec(
                name="opportunity_contact",
                numeric_columns=selected_numeric_columns(OPPORTUNITY_CONTACT_COLUMNS, available),
                boolean_columns=["confirmed_lineup", "road_game", "division_matchup", "doubleheader"],
                include_missing_indicators=True,
                include_booleans=True,
                include_categories=False,
                include_stop_counts=False,
                include_stop_reasons=False,
                balanced_loss=False,
                iterations=340,
                learning_rate=0.055,
                l2=0.03,
            ),
        ),
        Experiment(
            name="iteration_4_streak_safe_ranker",
            description="Hand-tuned high-floor ranker: opportunity/contact/pitcher vulnerability blend with only light stop penalties.",
            score_func=ranker_scores,
        ),
        Experiment(
            name="iteration_5_indicator_ranker",
            description="Hand-tuned ranker using the strongest observed historical indicators, with no stop-valve inputs.",
            score_func=indicator_ranker_scores,
        ),
        Experiment(
            name="iteration_6_components_plus_floor",
            description="Compact logistic: Bob score, component subscores, and the strongest opportunity/contact indicators.",
            matrix_spec=MatrixSpec(
                name="components_plus_floor",
                numeric_columns=selected_numeric_columns(
                    [
                        "score",
                        *COMPONENT_COLUMNS,
                        "hitter_pa_per_game_season",
                        "expected_pa",
                        "lineup_slot",
                        "hitter_last_5_games_ab",
                        "hitter_ba_500_ab",
                        "hitter_ba_2500_ab",
                        "hitter_hipa_500_pa",
                        "hitter_hipa_2500_pa",
                        "pitcher_hpi_season",
                        "inferred_pitch_type_ba",
                    ],
                    available,
                ),
                boolean_columns=["confirmed_lineup", "road_game", "division_matchup", "doubleheader"],
                include_missing_indicators=True,
                include_booleans=True,
                include_categories=False,
                include_stop_counts=False,
                include_stop_reasons=False,
                balanced_loss=False,
                iterations=320,
                learning_rate=0.055,
                l2=0.035,
            ),
        ),
    ]

    if args.experiments:
        requested = {value.strip() for value in args.experiments.split(",") if value.strip()}
        known = {experiment.name for experiment in experiments}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown experiment name(s): {', '.join(unknown)}")
        experiments = [experiment for experiment in experiments if experiment.name in requested]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_daily = []
    for experiment in experiments:
        print(f"Running {experiment.name}...", flush=True)
        summary, daily = evaluate_experiment(df, experiment, min_train_dates=args.min_train_dates)
        print(
            f"  {summary['wins']}-{summary['losses']} "
            f"({summary['daily_success_rate']:.3f}), max streak {summary['max_success_streak']}",
            flush=True,
        )
        summaries.append(summary)
        all_daily.extend(daily)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidates": str(Path(args.candidates).resolve()),
        "min_train_dates": args.min_train_dates,
        "rows": int(len(df)),
        "labeled_rows": int(df["label"].notna().sum()),
        "labeled_dates": int(df[df["label"].notna()]["date"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "summaries": summaries,
        "top_univariate_indicators": feature_univariate_summary(df)[:25],
        "stop_reason_hit_rates": stop_reason_summary(df),
    }
    summary_path = out_dir / f"learned_model_backtest_{timestamp}.json"
    daily_path = out_dir / f"learned_model_backtest_daily_{timestamp}.csv"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(daily_path, all_daily)
    print()
    print(f"Summary: {summary_path.resolve()}")
    print(f"Daily picks: {daily_path.resolve()}")


if __name__ == "__main__":
    main()
