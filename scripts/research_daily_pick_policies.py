from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest_learned_model_experiments import (
    MatrixBuilder,
    MatrixSpec,
    OPPORTUNITY_CONTACT_COLUMNS,
    fit_logistic,
    max_success_streak,
    prepare_dataframe,
    selected_numeric_columns,
)


DEFAULT_CANDIDATES = Path("data/statbirt_candidates.csv")
DEFAULT_OUT_DIR = Path("logs/model_research")
META_POLICY_NAME = "v3_top5_meta_selector"
META_FEATURES = [
    "model_score",
    "model_rank",
    "score_gap_to_top",
    "bob_score",
    "expected_pa",
    "lineup_slot",
    "pa_per_game",
    "starts_last_5",
    "hitter_last_5_games_ab",
    "hitter_last_5_games_ba",
    "hitter_ba_500_ab",
    "hitter_hipa_500_pa",
    "hitter_k_rate_500_pa",
    "hitter_whiff_rate_500_pa",
    "pitcher_hpi_season",
    "pitcher_stuff_plus",
    "pitcher_lr_opp_ba",
    "h2h_xba",
    "park_hit_factor",
    "stop_count",
    "confirmed_lineup",
]


def num(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(column, "")]), errors="coerce").iloc[0]
    return default if pd.isna(value) else float(value)


def stop_count(row: pd.Series) -> int:
    value = str(row.get("hard_pass_reasons", "") or "").strip()
    return 0 if not value else len([part for part in value.split("|") if part.strip()])


def is_success(row: pd.Series) -> int:
    return 1 if str(row.get("result_hit", "")).strip() == "1" else 0


def day_z(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    median = values.median(skipna=True)
    if pd.isna(median):
        median = 0.0
    filled = values.fillna(median)
    std = filled.std(ddof=0)
    if pd.isna(std) or std < 1e-8:
        std = 1.0
    return (filled - filled.mean()) / std


def daily_safety_score(rows: pd.DataFrame) -> pd.Series:
    return (
        rows["_model_score"]
        + 0.10 * day_z(rows["expected_pa"])
        + 0.10 * day_z(rows["hitter_pa_per_game_season"])
        - 0.08 * day_z(rows["lineup_slot"])
        + 0.06 * day_z(rows["hitter_ba_500_ab"])
        + 0.05 * day_z(rows["hitter_hipa_500_pa"])
        - 0.06 * day_z(rows["hitter_k_rate_500_pa"])
        - 0.04 * day_z(rows["pitcher_stuff_plus"])
        - 0.035 * rows["_stop_count"].astype(float)
    )


def meta_feature_rows(rows: pd.DataFrame) -> pd.DataFrame:
    ranked = rows.sort_values("_model_score", ascending=False).copy()
    top_score = float(ranked["_model_score"].iloc[0]) if len(ranked) else 0.0
    output = pd.DataFrame(index=ranked.index)
    output["model_score"] = pd.to_numeric(ranked["_model_score"], errors="coerce")
    output["model_rank"] = np.arange(1, len(ranked) + 1, dtype=float)
    output["score_gap_to_top"] = output["model_score"] - top_score
    output["bob_score"] = pd.to_numeric(ranked["score"], errors="coerce")
    output["expected_pa"] = pd.to_numeric(ranked["expected_pa"], errors="coerce")
    output["lineup_slot"] = pd.to_numeric(ranked["lineup_slot"], errors="coerce")
    output["pa_per_game"] = pd.to_numeric(ranked["hitter_pa_per_game_season"], errors="coerce")
    output["starts_last_5"] = pd.to_numeric(ranked["starts_last_5"], errors="coerce")
    output["hitter_last_5_games_ab"] = pd.to_numeric(ranked["hitter_last_5_games_ab"], errors="coerce")
    output["hitter_last_5_games_ba"] = pd.to_numeric(ranked["hitter_last_5_games_ba"], errors="coerce")
    output["hitter_ba_500_ab"] = pd.to_numeric(ranked["hitter_ba_500_ab"], errors="coerce")
    output["hitter_hipa_500_pa"] = pd.to_numeric(ranked["hitter_hipa_500_pa"], errors="coerce")
    output["hitter_k_rate_500_pa"] = pd.to_numeric(ranked["hitter_k_rate_500_pa"], errors="coerce")
    output["hitter_whiff_rate_500_pa"] = pd.to_numeric(ranked["hitter_whiff_rate_500_pa"], errors="coerce")
    output["pitcher_hpi_season"] = pd.to_numeric(ranked["pitcher_hpi_season"], errors="coerce")
    output["pitcher_stuff_plus"] = pd.to_numeric(ranked["pitcher_stuff_plus"], errors="coerce")
    output["pitcher_lr_opp_ba"] = pd.to_numeric(ranked["pitcher_lr_opp_ba"], errors="coerce")
    output["h2h_xba"] = pd.to_numeric(ranked["h2h_xba"], errors="coerce")
    output["park_hit_factor"] = pd.to_numeric(ranked["park_hit_factor"], errors="coerce")
    output["stop_count"] = ranked["_stop_count"].astype(float)
    output["confirmed_lineup"] = ranked.get("confirmed_lineup", "").map(lambda value: 1.0 if str(value).upper() == "Y" else 0.0)
    return output


def train_meta_predict(history: list[dict], current: pd.DataFrame) -> np.ndarray | None:
    if len(history) < 150:
        return None
    train = pd.DataFrame(history)
    if train["meta_label"].nunique() < 2:
        return None
    x_train_df = train[META_FEATURES].apply(pd.to_numeric, errors="coerce")
    current_features = meta_feature_rows(current)[META_FEATURES]
    medians = x_train_df.median(axis=0, skipna=True).fillna(0.0)
    x_train_df = x_train_df.fillna(medians)
    current_features = current_features.fillna(medians)
    means = x_train_df.mean(axis=0)
    stds = x_train_df.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    x_train = ((x_train_df - means) / stds).to_numpy(float)
    x_current = ((current_features - means) / stds).to_numpy(float)
    y_train = train["meta_label"].to_numpy(float)
    weights, intercept = fit_logistic(
        x_train,
        y_train,
        iterations=260,
        learning_rate=0.055,
        l2=0.04,
        balanced_loss=False,
    )
    return 1.0 / (1.0 + np.exp(-np.clip(x_current @ weights + intercept, -35.0, 35.0)))


@dataclass
class Policy:
    name: str
    description: str
    selector: Callable[[pd.DataFrame], pd.Series]


def choose_first(rows: pd.DataFrame) -> pd.Series:
    return rows.sort_values("_model_score", ascending=False).iloc[0]


def choose_top_graded(rows: pd.DataFrame) -> pd.Series:
    graded = rows[rows["label"].notna()]
    if graded.empty:
        return choose_first(rows)
    return graded.sort_values("_model_score", ascending=False).iloc[0]


def choose_pa_floor(rows: pd.DataFrame) -> pd.Series:
    eligible = rows[
        (pd.to_numeric(rows["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
        & (pd.to_numeric(rows["expected_pa"], errors="coerce") >= 4.1)
    ]
    return choose_first(eligible if not eligible.empty else rows)


def choose_lineup_floor(rows: pd.DataFrame) -> pd.Series:
    eligible = rows[
        (pd.to_numeric(rows["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
        & (pd.to_numeric(rows["expected_pa"], errors="coerce") >= 4.1)
        & (pd.to_numeric(rows["lineup_slot"], errors="coerce") <= 6)
    ]
    return choose_first(eligible if not eligible.empty else rows)


def choose_low_stop_floor(rows: pd.DataFrame) -> pd.Series:
    eligible = rows[
        (rows["_stop_count"] <= 6)
        & (pd.to_numeric(rows["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
        & (pd.to_numeric(rows["expected_pa"], errors="coerce") >= 4.0)
    ]
    return choose_first(eligible if not eligible.empty else rows)


def choose_top5_safety(rows: pd.DataFrame) -> pd.Series:
    top = rows.sort_values("_model_score", ascending=False).head(5).copy()
    top["_safety_score"] = daily_safety_score(top)
    return top.sort_values("_safety_score", ascending=False).iloc[0]


def choose_top8_safety_with_floor(rows: pd.DataFrame) -> pd.Series:
    top = rows.sort_values("_model_score", ascending=False).head(8).copy()
    eligible = top[
        (pd.to_numeric(top["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
        & (pd.to_numeric(top["expected_pa"], errors="coerce") >= 4.0)
        & (top["_stop_count"] <= 7)
    ].copy()
    pool = eligible if not eligible.empty else top
    pool["_safety_score"] = daily_safety_score(pool)
    return pool.sort_values("_safety_score", ascending=False).iloc[0]


def choose_bob_score_floor(rows: pd.DataFrame) -> pd.Series:
    eligible = rows[
        (pd.to_numeric(rows["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
        & (pd.to_numeric(rows["expected_pa"], errors="coerce") >= 4.0)
    ]
    pool = eligible if not eligible.empty else rows
    return pool.sort_values("score", ascending=False).iloc[0]


POLICIES = [
    Policy("v2_raw_top", "Highest walk-forward opportunity/contact probability.", choose_first),
    Policy("v2_top_graded_oracle", "Highest graded row only; diagnostic ceiling, not deployable.", choose_top_graded),
    Policy("v2_pa_floor", "Highest probability after PA/G >= 4.2 and expected PA >= 4.1.", choose_pa_floor),
    Policy("v2_lineup_floor", "PA floor plus lineup slot <= 6.", choose_lineup_floor),
    Policy("v2_low_stop_floor", "PA floor plus stop-valve count <= 6.", choose_low_stop_floor),
    Policy("v2_top5_safety", "Rerank top 5 by opportunity/contact safety score.", choose_top5_safety),
    Policy("v2_top8_safety_floor", "Rerank top 8 after PA and stop-count floor when available.", choose_top8_safety_with_floor),
    Policy("bob_score_pa_floor", "Highest Bob score after PA/G >= 4.2 and expected PA >= 4.0.", choose_bob_score_floor),
]


def streak_details(labels: list[int], dates: list[str]) -> dict:
    current = 0
    current_start = ""
    best = {"length": 0, "start": "", "end": ""}
    for label, date_value in zip(labels, dates, strict=True):
        if label == 1:
            if current == 0:
                current_start = date_value
            current += 1
            if current > best["length"]:
                best = {"length": current, "start": current_start, "end": date_value}
        else:
            current = 0
            current_start = ""
    return best


def summarize(rows: list[dict]) -> dict:
    labels = [int(row["hit"]) for row in rows]
    dates = [row["date"] for row in rows]
    statuses: dict[str, int] = {}
    years: dict[str, list[int]] = {}
    for row in rows:
        statuses[row["result_status"]] = statuses.get(row["result_status"], 0) + 1
        years.setdefault(row["date"][:4], []).append(int(row["hit"]))
    by_year = {
        year: {
            "dates": len(values),
            "wins": sum(values),
            "losses": len(values) - sum(values),
            "success_rate": sum(values) / len(values) if values else None,
        }
        for year, values in sorted(years.items())
    }
    return {
        "dates": len(rows),
        "wins": sum(labels),
        "losses": len(labels) - sum(labels),
        "success_rate": sum(labels) / len(labels) if labels else None,
        "max_success_streak": max_success_streak(labels),
        "best_streak": streak_details(labels, dates),
        "status_counts": dict(sorted(statuses.items())),
        "by_year": by_year,
    }


def topk_rescue_metrics(scored_rows_by_date: dict[str, pd.DataFrame]) -> dict:
    output = {}
    for k in (2, 3, 5, 8, 10):
        rescue = []
        for rows in scored_rows_by_date.values():
            top = rows.sort_values("_model_score", ascending=False).head(k)
            rescue.append(1 if (top["result_hit"].astype(str) == "1").any() else 0)
        output[f"top_{k}_contains_hit"] = {
            "wins": int(sum(rescue)),
            "dates": len(rescue),
            "rate": float(sum(rescue) / len(rescue)) if rescue else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Research daily one-pick selection policies.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-train-dates", type=int, default=30)
    args = parser.parse_args()

    df = prepare_dataframe(Path(args.candidates))
    available = set(df.columns)
    spec = MatrixSpec(
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
    )
    builder = MatrixBuilder(df, spec)
    labeled = df[df["label"].notna()].copy()
    dates = sorted(labeled["date"].unique())

    policy_rows: dict[str, list[dict]] = {policy.name: [] for policy in POLICIES}
    policy_rows[META_POLICY_NAME] = []
    scored_by_date: dict[str, pd.DataFrame] = {}
    meta_history: list[dict] = []

    for date_value in dates[args.min_train_dates :]:
        train_idx = df.index[(df["date"] < date_value) & df["label"].notna()].to_numpy()
        test_idx = df.index[df["date"].eq(date_value)].to_numpy()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        x_train, x_test = builder.train_test_arrays(train_idx, test_idx)
        y_train = df.loc[train_idx, "label"].to_numpy(float)
        weights, intercept = fit_logistic(
            x_train,
            y_train,
            iterations=spec.iterations,
            learning_rate=spec.learning_rate,
            l2=spec.l2,
            balanced_loss=spec.balanced_loss,
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(x_test @ weights + intercept, -35.0, 35.0)))
        rows = df.loc[test_idx].copy()
        rows["_model_score"] = probability
        rows["_stop_count"] = rows.apply(stop_count, axis=1)
        rows = rows.sort_values("_model_score", ascending=False).copy()
        rows["_model_rank"] = np.arange(1, len(rows) + 1)
        scored_by_date[date_value] = rows

        for policy in POLICIES:
            pick = policy.selector(rows)
            policy_rows[policy.name].append(
                {
                    "policy": policy.name,
                    "date": date_value,
                    "player": pick.get("player", ""),
                    "team": pick.get("team", ""),
                    "opponent": pick.get("opponent", ""),
                    "hit": is_success(pick),
                    "result_hit": pick.get("result_hit", ""),
                    "result_status": pick.get("result_status", ""),
                    "model_score": float(pick.get("_model_score", 0.0)),
                    "bob_score": num(pick, "score"),
                    "expected_pa": num(pick, "expected_pa"),
                    "lineup_slot": num(pick, "lineup_slot"),
                    "pa_per_game": num(pick, "hitter_pa_per_game_season"),
                    "stop_count": int(pick.get("_stop_count", 0)),
                    "pickable": pick.get("pickable", ""),
                }
            )

        top5 = rows.head(5).copy()
        meta_probability = train_meta_predict(meta_history, top5)
        if meta_probability is None:
            meta_pick = top5.iloc[0]
            meta_score = float(meta_pick.get("_model_score", 0.0))
        else:
            top5 = top5.copy()
            top5["_meta_score"] = meta_probability
            meta_pick = top5.sort_values("_meta_score", ascending=False).iloc[0]
            meta_score = float(meta_pick["_meta_score"])
        policy_rows[META_POLICY_NAME].append(
            {
                "policy": META_POLICY_NAME,
                "date": date_value,
                "player": meta_pick.get("player", ""),
                "team": meta_pick.get("team", ""),
                "opponent": meta_pick.get("opponent", ""),
                "hit": is_success(meta_pick),
                "result_hit": meta_pick.get("result_hit", ""),
                "result_status": meta_pick.get("result_status", ""),
                "model_score": float(meta_pick.get("_model_score", 0.0)),
                "bob_score": num(meta_pick, "score"),
                "expected_pa": num(meta_pick, "expected_pa"),
                "lineup_slot": num(meta_pick, "lineup_slot"),
                "pa_per_game": num(meta_pick, "hitter_pa_per_game_season"),
                "stop_count": int(meta_pick.get("_stop_count", 0)),
                "pickable": meta_pick.get("pickable", ""),
                "meta_score": meta_score,
            }
        )

        meta_features = meta_feature_rows(top5)
        for idx, feature_row in meta_features.iterrows():
            record = {feature: float(feature_row.get(feature, 0.0)) for feature in META_FEATURES}
            record["meta_label"] = is_success(top5.loc[idx])
            meta_history.append(record)

    summaries = []
    for policy in POLICIES:
        summary = summarize(policy_rows[policy.name])
        summary.update({"policy": policy.name, "description": policy.description})
        summaries.append(summary)
    meta_summary = summarize(policy_rows[META_POLICY_NAME])
    meta_summary.update(
        {
            "policy": META_POLICY_NAME,
            "description": "Second-stage walk-forward meta-model chooses within the base model's top 5.",
        }
    )
    summaries.append(meta_summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"daily_pick_policy_research_{timestamp}.json"
    daily_path = out_dir / f"daily_pick_policy_research_daily_{timestamp}.csv"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidates": str(Path(args.candidates).resolve()),
        "min_train_dates": args.min_train_dates,
        "rows": int(len(df)),
        "labeled_rows": int(df["label"].notna().sum()),
        "labeled_dates": int(labeled["date"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "topk_rescue": topk_rescue_metrics(scored_by_date),
        "summaries": sorted(summaries, key=lambda row: row["success_rate"], reverse=True),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    all_rows = [row for rows in policy_rows.values() for row in rows]
    fieldnames = []
    for row in all_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with daily_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    for summary in payload["summaries"]:
        print(
            f"{summary['policy']}: {summary['wins']}-{summary['losses']} "
            f"({summary['success_rate']:.3f}), max streak {summary['max_success_streak']}"
        )
    print()
    print(f"Summary: {summary_path.resolve()}")
    print(f"Daily picks: {daily_path.resolve()}")


if __name__ == "__main__":
    main()
