from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_learned_model_experiments import (  # noqa: E402
    BASE_NUMERIC_COLUMNS,
    COMPONENT_COLUMNS,
    MatrixBuilder,
    MatrixSpec,
    OPPORTUNITY_CONTACT_COLUMNS,
    fit_logistic,
    max_success_streak,
    prepare_dataframe,
    selected_numeric_columns,
    split_pipe,
)
from scripts.research_daily_pick_policies import daily_safety_score  # noqa: E402
from statbirt.export_web import DEFAULT_CONGREGATION_CSV, congregation_record_for, load_congregation  # noqa: E402
from statbirt.utils import parse_float, parse_int  # noqa: E402


DEFAULT_CANDIDATES = Path("data/statbirt_candidates.csv")
DEFAULT_MODEL_JSON = Path("data/models/selection_strategy_report.json")
DEFAULT_WEB_JSON = Path("web/data/selection_strategy_report.json")
DEFAULT_WEB_HTML = Path("web/data/selection_strategy_report.html")
DEFAULT_REPORT_DIR = Path("reports/selection_strategy")
DEFAULT_WEB_REPORT = Path("reports/selection_strategy_latest.html")

NUMERIC_RESEARCH_COLUMNS = [
    "score",
    "expected_pa",
    "lineup_slot",
    "starts_last_5",
    "hitter_last_5_games_played",
    "hitter_last_5_games_hits",
    "hitter_last_5_games_ab",
    "hitter_last_5_games_ba",
    "hitter_pa_per_game_season",
    "hitter_ba_season",
    "hitter_ba_2500_ab",
    "hitter_ba_500_ab",
    "hitter_ba_75_ab",
    "hitter_ba_25_ab",
    "hitter_hipa_2500_pa",
    "hitter_hipa_500_pa",
    "hitter_hipa_75_ab",
    "hitter_bb_rate_season",
    "hitter_bb_rate_500_pa",
    "hitter_whiff_rate_season",
    "hitter_whiff_rate_500_pa",
    "hitter_k_rate_season",
    "hitter_k_rate_500_pa",
    "hitter_split_ba_season_vs_lhp",
    "hitter_split_ba_season_vs_rhp",
    "hitter_split_pa_season_vs_lhp",
    "hitter_split_pa_season_vs_rhp",
    "hitter_split_ba_500_vs_lhp",
    "hitter_split_ba_500_vs_rhp",
    "hitter_split_ba_1500_vs_lhp",
    "hitter_split_ba_1500_vs_rhp",
    "pitcher_hpi_350",
    "pitcher_hpi_200",
    "pitcher_hpi_season",
    "pitcher_hits_last_18_ip",
    "pitcher_stuff_plus",
    "pitcher_last_start_ip",
    "pitcher_last_start_hits",
    "pitcher_last_start_strikeouts",
    "pitcher_last_start_walks",
    "h2h_pa",
    "h2h_hits",
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

SCORE_BUCKETS = [
    ("<= 50", -math.inf, 50.0, True),
    ("50-55", 50.0, 55.0, False),
    ("55-60", 55.0, 60.0, False),
    ("60-65", 60.0, 65.0, False),
    ("65-70", 65.0, 70.0, False),
    ("70-75", 70.0, 75.0, False),
    ("75-80", 75.0, 80.0, False),
    (">= 80", 80.0, math.inf, True),
]


def boolish(value: object) -> bool:
    return str(value or "").strip().lower() in {"y", "yes", "true", "1"}


def explicit_label(value: object) -> int | None:
    text = str(value or "").strip()
    if text == "1":
        return 1
    if text == "0":
        return 0
    return None


def rate(hits: int, total: int) -> float | None:
    return None if total <= 0 else hits / total


def format_rate(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{value * 100:.{digits}f}%"


def format_float(value: object, digits: int = 3) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "--"
    return f"{parsed:.{digits}f}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_engineered_columns(df: pd.DataFrame, congregation_csv: Path) -> pd.DataFrame:
    df = df.copy()
    for column in set(NUMERIC_RESEARCH_COLUMNS + BASE_NUMERIC_COLUMNS + COMPONENT_COLUMNS + OPPORTUNITY_CONTACT_COLUMNS):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["_explicit_label"] = df.get("result_hit", "").map(explicit_label)
    df["_stop_count"] = df.get("hard_pass_reasons", "").map(lambda value: len(split_pipe(value)))
    df["_pickable"] = df.get("pickable", "").map(boolish)
    df["_confirmed_lineup"] = df.get("confirmed_lineup", "").map(boolish)
    df["_h2h_seen"] = pd.to_numeric(df.get("h2h_pa", 0), errors="coerce").fillna(0) > 0
    df["_h2h_3pa_plus"] = pd.to_numeric(df.get("h2h_pa", 0), errors="coerce").fillna(0) >= 3
    df["_hot_400_last5"] = (
        (pd.to_numeric(df.get("hitter_last_5_games_ba", np.nan), errors="coerce") >= 0.400)
        & (pd.to_numeric(df.get("hitter_last_5_games_ab", 0), errors="coerce").fillna(0) > 0)
    )
    df["_volume_floor"] = (
        (pd.to_numeric(df.get("expected_pa", np.nan), errors="coerce") >= 4.1)
        & (pd.to_numeric(df.get("hitter_pa_per_game_season", np.nan), errors="coerce") >= 4.2)
    )
    df["_low_k_rate"] = pd.to_numeric(df.get("hitter_k_rate_500_pa", np.nan), errors="coerce") <= 0.20
    df["_season_ba_275"] = pd.to_numeric(df.get("hitter_ba_season", np.nan), errors="coerce") >= 0.275
    df["_season_ba_300"] = pd.to_numeric(df.get("hitter_ba_season", np.nan), errors="coerce") >= 0.300
    df["_dominant_start_stop"] = df.get("hard_pass_reasons", "").astype(str).str.lower().str.contains("dominant start")
    df["_no_stop_valves"] = df["_stop_count"].eq(0)

    congregation = load_congregation(congregation_csv)
    publisher_flags = []
    for row in df[["player", "player_id"]].fillna("").to_dict("records"):
        record = congregation_record_for(row, congregation)
        publisher_flags.append(str((record or {}).get("status") or "").strip().lower() == "publisher")
    df["_congregation_publisher"] = publisher_flags
    return df


def overview_summary(df: pd.DataFrame) -> dict:
    labeled = df[df["_explicit_label"].notna()]
    by_year = {}
    for year, rows in labeled.groupby(labeled["date"].astype(str).str.slice(0, 4)):
        hits = int(rows["_explicit_label"].sum())
        total = int(len(rows))
        by_year[year] = {
            "decisions": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": rate(hits, total),
        }
    return {
        "candidate_rows": int(len(df)),
        "labeled_decisions": int(len(labeled)),
        "labeled_dates": int(labeled["date"].nunique()),
        "date_min": str(labeled["date"].min()) if len(labeled) else "",
        "date_max": str(labeled["date"].max()) if len(labeled) else "",
        "overall_hit_rate": rate(int(labeled["_explicit_label"].sum()), int(len(labeled))),
        "by_year": by_year,
    }


def score_bucket_summary(df: pd.DataFrame) -> list[dict]:
    labeled = df[df["_explicit_label"].notna()].copy()
    output = []
    scores = pd.to_numeric(labeled.get("score", np.nan), errors="coerce")
    for label, low, high, closed_boundary in SCORE_BUCKETS:
        if math.isinf(low):
            mask = scores <= high
        elif math.isinf(high):
            mask = scores >= low
        elif closed_boundary:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores > low) & (scores <= high)
        rows = labeled[mask].copy()
        hits = int(rows["_explicit_label"].sum())
        count = int(len(rows))
        output.append(
            {
                "score_range": label,
                "decisions": count,
                "hits": hits,
                "misses": count - hits,
                "hit_rate": rate(hits, count),
                "avg_score": None if count == 0 else float(pd.to_numeric(rows["score"], errors="coerce").mean()),
                "pickable_rate": rate(int(rows["_pickable"].sum()), count),
                "publisher_rate": rate(int(rows["_congregation_publisher"].sum()), count),
                "avg_stop_count": None if count == 0 else float(rows["_stop_count"].mean()),
            }
        )
    return output


def binary_factor_summary(df: pd.DataFrame) -> list[dict]:
    factors = [
        ("Pickable", "_pickable"),
        ("Publisher / congregation", "_congregation_publisher"),
        ("Confirmed lineup", "_confirmed_lineup"),
        ("No stop valves", "_no_stop_valves"),
        ("Volume floor (PA/G and expected PA)", "_volume_floor"),
        ("H2H seen starter", "_h2h_seen"),
        ("H2H 3+ PA", "_h2h_3pa_plus"),
        ("Hot .400+ last 5", "_hot_400_last5"),
        ("Season BA >= .275", "_season_ba_275"),
        ("Season BA >= .300", "_season_ba_300"),
        ("K rate <= 20%", "_low_k_rate"),
        ("Dominant-start stop valve", "_dominant_start_stop"),
        ("Road game", "bool__road_game"),
        ("Division matchup", "bool__division_matchup"),
        ("Doubleheader", "bool__doubleheader"),
    ]
    labeled = df[df["_explicit_label"].notna()].copy()
    output = []
    for label, column in factors:
        if column not in labeled.columns:
            continue
        yes_mask = labeled[column].fillna(False).astype(bool)
        yes = labeled[yes_mask]
        no = labeled[~yes_mask]
        if len(yes) < 25:
            continue
        yes_hits = int(yes["_explicit_label"].sum())
        no_hits = int(no["_explicit_label"].sum())
        yes_rate = rate(yes_hits, int(len(yes)))
        no_rate = rate(no_hits, int(len(no)))
        output.append(
            {
                "factor": label,
                "yes_decisions": int(len(yes)),
                "yes_hits": yes_hits,
                "yes_hit_rate": yes_rate,
                "no_decisions": int(len(no)),
                "no_hits": no_hits,
                "no_hit_rate": no_rate,
                "edge_vs_no": None if yes_rate is None or no_rate is None else yes_rate - no_rate,
            }
        )
    return sorted(output, key=lambda row: abs(row.get("edge_vs_no") or 0.0), reverse=True)


def numeric_feature_summary(df: pd.DataFrame, min_rows: int = 300) -> list[dict]:
    labeled = df[df["_explicit_label"].notna()].copy()
    output = []
    for column in selected_numeric_columns(NUMERIC_RESEARCH_COLUMNS, set(labeled.columns)):
        values = pd.to_numeric(labeled[column], errors="coerce")
        usable = labeled[values.notna()].copy()
        if len(usable) < min_rows or usable["_explicit_label"].nunique() < 2:
            continue
        usable["_feature"] = pd.to_numeric(usable[column], errors="coerce")
        corr = float(usable["_feature"].corr(usable["_explicit_label"])) if usable["_feature"].std(ddof=0) > 0 else 0.0
        try:
            bins = pd.qcut(usable["_feature"], q=5, duplicates="drop")
            grouped = usable.groupby(bins, observed=True)["_explicit_label"].agg(["count", "sum", "mean"])
            low = grouped.iloc[0]
            high = grouped.iloc[-1]
            low_rate = float(low["mean"])
            high_rate = float(high["mean"])
            low_count = int(low["count"])
            high_count = int(high["count"])
        except Exception:
            low_rate = high_rate = None
            low_count = high_count = 0
        output.append(
            {
                "feature": column,
                "decisions": int(len(usable)),
                "correlation": corr,
                "bottom_quintile_decisions": low_count,
                "bottom_quintile_hit_rate": low_rate,
                "top_quintile_decisions": high_count,
                "top_quintile_hit_rate": high_rate,
                "top_minus_bottom": None if low_rate is None or high_rate is None else high_rate - low_rate,
            }
        )
    return sorted(output, key=lambda row: abs(row.get("top_minus_bottom") or 0.0), reverse=True)


def stop_reason_summary(df: pd.DataFrame) -> list[dict]:
    labeled = df[df["_explicit_label"].notna()].copy()
    grouped: dict[str, list[int]] = {"No stop valve": []}
    for _, row in labeled.iterrows():
        reasons = split_pipe(row.get("hard_pass_reasons", ""))
        label = int(row["_explicit_label"])
        if not reasons:
            grouped["No stop valve"].append(label)
        for reason in reasons:
            grouped.setdefault(reason, []).append(label)
    output = []
    overall = rate(int(labeled["_explicit_label"].sum()), int(len(labeled))) or 0.0
    for reason, labels in grouped.items():
        if len(labels) < 25:
            continue
        reason_rate = rate(sum(labels), len(labels))
        output.append(
            {
                "stop_reason": reason,
                "decisions": len(labels),
                "hits": int(sum(labels)),
                "hit_rate": reason_rate,
                "edge_vs_overall": None if reason_rate is None else reason_rate - overall,
            }
        )
    return sorted(output, key=lambda row: row["hit_rate"] or 0.0, reverse=True)


def row_hit(row: pd.Series) -> int | None:
    return explicit_label(row.get("result_hit", ""))


def strategy_ranked_rows(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    ranked = rows.copy()
    if ranked.empty:
        return ranked
    if name == "wf_probability_top":
        return ranked.sort_values("_model_score", ascending=False)
    if name == "wf_volume_floor":
        eligible = ranked[
            (pd.to_numeric(ranked["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
            & (pd.to_numeric(ranked["expected_pa"], errors="coerce") >= 4.1)
        ]
        pool = eligible if not eligible.empty else ranked
        return pool.sort_values("_model_score", ascending=False)
    if name == "wf_low_stop_floor":
        eligible = ranked[
            (ranked["_stop_count"] <= 6)
            & (pd.to_numeric(ranked["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
            & (pd.to_numeric(ranked["expected_pa"], errors="coerce") >= 4.0)
        ]
        pool = eligible if not eligible.empty else ranked
        return pool.sort_values("_model_score", ascending=False)
    if name == "wf_top5_safety_blend":
        pool = ranked.sort_values("_model_score", ascending=False).head(5).copy()
        pool["_strategy_score"] = daily_safety_score(pool)
        return pool.sort_values("_strategy_score", ascending=False)
    if name == "wf_top8_safety_floor":
        top = ranked.sort_values("_model_score", ascending=False).head(8).copy()
        eligible = top[
            (pd.to_numeric(top["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
            & (pd.to_numeric(top["expected_pa"], errors="coerce") >= 4.0)
            & (top["_stop_count"] <= 7)
        ].copy()
        pool = eligible if not eligible.empty else top
        pool["_strategy_score"] = daily_safety_score(pool)
        return pool.sort_values("_strategy_score", ascending=False)
    if name == "bob_score_volume_floor":
        eligible = ranked[
            (pd.to_numeric(ranked["hitter_pa_per_game_season"], errors="coerce") >= 4.2)
            & (pd.to_numeric(ranked["expected_pa"], errors="coerce") >= 4.0)
        ]
        pool = eligible if not eligible.empty else ranked
        return pool.sort_values("score", ascending=False)
    raise ValueError(f"Unknown strategy: {name}")


STRATEGIES = [
    {
        "name": "wf_probability_top",
        "label": "Walk-forward probability top",
        "description": "Select the highest probability hitter from the walk-forward opportunity/contact model.",
    },
    {
        "name": "wf_volume_floor",
        "label": "Probability with volume floor",
        "description": "Prefer hitters with PA/G >= 4.2 and expected PA >= 4.1, then sort by walk-forward probability.",
    },
    {
        "name": "wf_low_stop_floor",
        "label": "Probability with volume + stop floor",
        "description": "Apply the volume floor and require six or fewer stop valves when possible.",
    },
    {
        "name": "wf_top5_safety_blend",
        "label": "Top-5 safety blend",
        "description": "Take the model top 5, then rerank by opportunity/contact safety factors.",
    },
    {
        "name": "wf_top8_safety_floor",
        "label": "Top-8 safety blend with floor",
        "description": "Take the model top 8, apply PA and stop-count floors when possible, then rerank by safety.",
    },
    {
        "name": "bob_score_volume_floor",
        "label": "Bob score with volume floor",
        "description": "Primary Bob score baseline after a PA volume floor.",
    },
]


def summarize_strategy(rows: list[dict]) -> dict:
    top1 = [row for row in rows if row["top1_decision"]]
    top2 = [row for row in rows if row["top2_decision"]]
    top1_values = [1 if row["top1_hit"] else 0 for row in top1]
    top2_any_values = [1 if row["top2_any_hit"] else 0 for row in top2]
    top2_both_values = [1 if row["top2_both_hit"] else 0 for row in top2]
    years: dict[str, list[int]] = {}
    for row in top1:
        years.setdefault(str(row["date"])[:4], []).append(1 if row["top1_hit"] else 0)
    by_year = {
        year: {
            "top1_decisions": len(values),
            "top1_wins": sum(values),
            "top1_hit_rate": rate(sum(values), len(values)),
        }
        for year, values in sorted(years.items())
    }
    return {
        "strategy": rows[0]["strategy"] if rows else "",
        "label": rows[0]["strategy_label"] if rows else "",
        "description": rows[0]["strategy_description"] if rows else "",
        "top1_decisions": len(top1),
        "top1_hits": sum(top1_values),
        "top1_misses": len(top1_values) - sum(top1_values),
        "top1_hit_rate": rate(sum(top1_values), len(top1_values)),
        "top1_max_streak": max_success_streak(top1_values),
        "top2_decisions": len(top2),
        "top2_any_hits": sum(top2_any_values),
        "top2_any_hit_rate": rate(sum(top2_any_values), len(top2_any_values)),
        "top2_any_max_streak": max_success_streak(top2_any_values),
        "top2_both_hits": sum(top2_both_values),
        "top2_both_hit_rate": rate(sum(top2_both_values), len(top2_both_values)),
        "ungraded_top1": len(rows) - len(top1),
        "by_year": by_year,
    }


def record_text(hits: int, total: int) -> str:
    return f"{hits} / {total}"


def publisher_split_summary(daily_rows: list[dict], strategy_name: str = "wf_probability_top") -> dict:
    rows = [row for row in daily_rows if row.get("strategy") == strategy_name]
    strategy_label = next((row.get("strategy_label") for row in rows if row.get("strategy_label")), strategy_name)

    def bool_summary(source_rows: list[dict], hit_key: str) -> dict:
        hits = sum(1 for row in source_rows if row.get(hit_key) is True)
        total = len(source_rows)
        return {
            "decisions": total,
            "hits": hits,
            "misses": total - hits,
            "record": record_text(hits, total),
            "hit_rate": rate(hits, total),
        }

    top1_rows = [row for row in rows if row.get("top1_decision")]
    top1_by_publisher = []
    for flag, label in ((True, "Publisher"), (False, "Non-publisher")):
        summary = bool_summary([row for row in top1_rows if bool(row.get("top1_publisher")) is flag], "top1_hit")
        summary["group"] = label
        top1_by_publisher.append(summary)

    pick_rows: list[dict] = []
    for row in rows:
        if not row.get("top2_decision"):
            continue
        for slot in ("top2_pick1", "top2_pick2"):
            hit = row.get(f"{slot}_hit")
            if hit is None:
                continue
            pick_rows.append(
                {
                    "publisher": bool(row.get(f"{slot}_publisher")),
                    "hit": hit == 1,
                }
            )

    top2_individual_by_publisher = []
    for flag, label in ((True, "Publisher picks"), (False, "Non-publisher picks")):
        summary = bool_summary([row for row in pick_rows if row["publisher"] is flag], "hit")
        summary["group"] = label
        top2_individual_by_publisher.append(summary)

    card_rows = [row for row in rows if row.get("top2_decision")]
    card_by_publisher_count = []
    for count in (0, 1, 2):
        group = [row for row in card_rows if int(row.get("top2_publisher_count") or 0) == count]
        any_hits = sum(1 for row in group if row.get("top2_any_hit") is True)
        both_hits = sum(1 for row in group if row.get("top2_both_hit") is True)
        total = len(group)
        card_by_publisher_count.append(
            {
                "group": f"{count} publishers",
                "days": total,
                "any_hit_record": record_text(any_hits, total),
                "any_hit_rate": rate(any_hits, total),
                "both_hit_record": record_text(both_hits, total),
                "both_hit_rate": rate(both_hits, total),
            }
        )

    card_has_publisher = []
    for flag, label in ((True, "At least one publisher"), (False, "No publishers")):
        group = [row for row in card_rows if (int(row.get("top2_publisher_count") or 0) > 0) is flag]
        any_hits = sum(1 for row in group if row.get("top2_any_hit") is True)
        both_hits = sum(1 for row in group if row.get("top2_both_hit") is True)
        total = len(group)
        card_has_publisher.append(
            {
                "group": label,
                "days": total,
                "any_hit_record": record_text(any_hits, total),
                "any_hit_rate": rate(any_hits, total),
                "both_hit_record": record_text(both_hits, total),
                "both_hit_rate": rate(both_hits, total),
            }
        )

    return {
        "strategy": strategy_name,
        "strategy_label": strategy_label,
        "top1_by_publisher": top1_by_publisher,
        "top2_individual_by_publisher": top2_individual_by_publisher,
        "top2_card_by_publisher_count": card_by_publisher_count,
        "top2_card_has_publisher": card_has_publisher,
    }


def run_walk_forward_strategy(df: pd.DataFrame, min_train_dates: int) -> tuple[list[dict], list[dict]]:
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
        iterations=300,
        learning_rate=0.055,
        l2=0.03,
    )
    builder = MatrixBuilder(df, spec)
    labeled = df[df["label"].notna()].copy()
    dates = sorted(labeled["date"].unique())
    daily_rows: list[dict] = []

    for date_value in dates[min_train_dates:]:
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
        day_rows = df.loc[test_idx].copy()
        day_rows["_model_score"] = probability
        day_rows["_stop_count"] = day_rows.get("hard_pass_reasons", "").map(lambda value: len(split_pipe(value)))

        for strategy in STRATEGIES:
            ranked = strategy_ranked_rows(strategy["name"], day_rows)
            if ranked.empty:
                continue
            pair = ranked.head(2)
            top1 = ranked.iloc[0]
            top1_result = row_hit(top1)
            pair_results = [row_hit(row) for _, row in pair.iterrows()]
            pair_records = [
                {
                    "player": str(row.get("player", "")),
                    "publisher": bool(row.get("_congregation_publisher", False)),
                    "hit": row_hit(row),
                }
                for _, row in pair.iterrows()
            ]
            top2_decision = len(pair_results) == 2 and all(value is not None for value in pair_results)
            daily_rows.append(
                {
                    "date": str(date_value),
                    "strategy": strategy["name"],
                    "strategy_label": strategy["label"],
                    "strategy_description": strategy["description"],
                    "top1_player": top1.get("player", ""),
                    "top1_team": top1.get("team", ""),
                    "top1_opponent": top1.get("opponent", ""),
                    "top1_probability": float(top1.get("_model_score", 0.0)),
                    "top1_bob_score": parse_float(top1.get("score")),
                    "top1_expected_pa": parse_float(top1.get("expected_pa")),
                    "top1_lineup_slot": parse_float(top1.get("lineup_slot")),
                    "top1_stop_count": int(top1.get("_stop_count", 0)),
                    "top1_pickable": top1.get("pickable", ""),
                    "top1_publisher": bool(top1.get("_congregation_publisher", False)),
                    "top1_result_hit": top1_result,
                    "top1_decision": top1_result is not None,
                    "top1_hit": top1_result == 1,
                    "top2_pair": " + ".join(str(row.get("player", "")) for _, row in pair.iterrows()),
                    "top2_publisher_count": sum(1 for record in pair_records if record["publisher"]),
                    "top2_pick1_player": pair_records[0]["player"] if len(pair_records) > 0 else "",
                    "top2_pick1_publisher": pair_records[0]["publisher"] if len(pair_records) > 0 else False,
                    "top2_pick1_hit": pair_records[0]["hit"] if len(pair_records) > 0 else None,
                    "top2_pick2_player": pair_records[1]["player"] if len(pair_records) > 1 else "",
                    "top2_pick2_publisher": pair_records[1]["publisher"] if len(pair_records) > 1 else False,
                    "top2_pick2_hit": pair_records[1]["hit"] if len(pair_records) > 1 else None,
                    "top2_decision": top2_decision,
                    "top2_any_hit": top2_decision and any(value == 1 for value in pair_results),
                    "top2_both_hit": top2_decision and all(value == 1 for value in pair_results),
                }
            )

    summary_rows = [summarize_strategy(rows) for _, rows in group_by(daily_rows, "strategy").items()]
    return sorted(summary_rows, key=lambda row: (row["top2_any_hit_rate"] or 0.0, row["top1_hit_rate"] or 0.0), reverse=True), daily_rows


def group_by(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def html_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    if not rows:
        return "<p class=\"muted\">No rows available.</p>"
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _, kind in columns:
            value = row.get(key)
            if kind == "percent":
                text = format_rate(value)
            elif kind == "float1":
                text = "--" if value is None else f"{float(value):.1f}"
            elif kind == "float3":
                text = "--" if value is None else f"{float(value):.3f}"
            elif kind == "signed_percent":
                text = "--" if value is None else f"{float(value) * 100:+.1f} pts"
            else:
                text = str(value if value is not None else "--")
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_html_report(
    *,
    payload: dict,
    score_rows: list[dict],
    binary_rows: list[dict],
    numeric_rows: list[dict],
    stop_rows: list[dict],
    strategy_rows: list[dict],
    publisher_split: dict,
) -> str:
    overview = payload["overview"]
    best_top1_strategy = payload.get("best_top1_strategy") or {}
    best_top2_strategy = payload.get("best_top2_strategy") or {}
    top_binary = binary_rows[:5]
    top_numeric = numeric_rows[:8]
    generated = payload["generated_at"]
    publisher_strategy_label = publisher_split.get("strategy_label") or "Walk-forward probability top"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Statbirt Selection Strategy Research</title>
  <style>
    :root {{ color-scheme: light; --bg:#f4f6f5; --surface:#fff; --ink:#17211f; --muted:#65716d; --line:#dce4e0; --good:#137b5f; --starter:#4169d8; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 44px; }}
    header, section {{ margin-bottom:16px; padding:18px; border:1px solid var(--line); border-radius:8px; background:var(--surface); }}
    h1 {{ margin:0; font-size:2rem; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:1.08rem; }}
    h3 {{ margin:16px 0 8px; font-size:.92rem; letter-spacing:0; }}
    p {{ margin:6px 0 0; color:var(--muted); line-height:1.45; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; margin-top:14px; }}
    .kpi {{ padding:12px; border:1px solid var(--line); border-radius:8px; background:#fbfcfc; }}
    .kpi span {{ display:block; color:var(--muted); font-size:.72rem; font-weight:800; text-transform:uppercase; }}
    .kpi strong {{ display:block; margin-top:4px; font-size:1.35rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:.84rem; }}
    th, td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#eef3f1; color:var(--muted); font-size:.7rem; text-transform:uppercase; }}
    tr:hover td {{ background:#fafbfb; }}
    .muted {{ color:var(--muted); }}
    .two {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    ul {{ margin:8px 0 0; padding-left:18px; }}
    li {{ margin:4px 0; color:#30403c; }}
    @media (max-width: 820px) {{ .kpis, .two {{ grid-template-columns:1fr; }} main {{ width:min(100vw - 20px, 760px); }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Statbirt Selection Strategy Research</h1>
      <p>Generated {html.escape(generated)} from {html.escape(payload["source_candidates"])}. Only rows with explicit result_hit values of 1 or 0 are counted as decisions.</p>
      <div class="kpis">
        <div class="kpi"><span>Labeled decisions</span><strong>{overview["labeled_decisions"]:,}</strong></div>
        <div class="kpi"><span>Game days</span><strong>{overview["labeled_dates"]:,}</strong></div>
        <div class="kpi"><span>Date range</span><strong>{html.escape(overview["date_min"])} to {html.escape(overview["date_max"])}</strong></div>
        <div class="kpi"><span>Overall hit rate</span><strong>{format_rate(overview["overall_hit_rate"])}</strong></div>
      </div>
    </header>

    <section>
      <h2>Executive Read</h2>
      <p>For a single pick, the best walk-forward policy in this run is <strong>{html.escape(str(best_top1_strategy.get("label", "--")))}</strong> at <strong>{format_rate(best_top1_strategy.get("top1_hit_rate"))}</strong>. For a two-pick card, the best any-hit policy is <strong>{html.escape(str(best_top2_strategy.get("label", "--")))}</strong> at <strong>{format_rate(best_top2_strategy.get("top2_any_hit_rate"))}</strong>.</p>
      <p>The analysis is intentionally split between broad factor discovery and deployable daily policy testing. Factor tables show correlation/cohort signal across all candidate decisions; strategy tables show what would have happened using only prior dates to fit the daily model.</p>
    </section>

    <section>
      <h2>Walk-Forward Daily Selection Strategies</h2>
      {html_table(strategy_rows, [
        ("label", "Strategy", "text"),
        ("top1_decisions", "Top-1 days", "text"),
        ("top1_hits", "Top-1 hits", "text"),
        ("top1_hit_rate", "Top-1 rate", "percent"),
        ("top1_max_streak", "Top-1 max streak", "text"),
        ("top2_decisions", "Top-2 days", "text"),
        ("top2_any_hit_rate", "Top-2 any hit", "percent"),
        ("top2_any_max_streak", "Top-2 max streak", "text"),
      ])}
    </section>

    <section>
      <h2>Publisher Split For Two-Pick Policy</h2>
      <p>This section breaks down the <strong>{html.escape(str(publisher_strategy_label))}</strong> policy by whether selected hitters were marked Publisher in congregation.csv. The two-pick any-hit metric means at least one of the two selected hitters got a hit that day.</p>
      <h3>Top-1 Pick By Publisher Status</h3>
      {html_table(publisher_split.get("top1_by_publisher", []), [
        ("group", "Group", "text"),
        ("record", "Record", "text"),
        ("hit_rate", "Hit rate", "percent"),
      ])}
      <h3>Individual Top-2 Picks By Publisher Status</h3>
      {html_table(publisher_split.get("top2_individual_by_publisher", []), [
        ("group", "Group", "text"),
        ("record", "Record", "text"),
        ("hit_rate", "Hit rate", "percent"),
      ])}
      <h3>Two-Pick Card By Publisher Count</h3>
      {html_table(publisher_split.get("top2_card_by_publisher_count", []), [
        ("group", "Pair type", "text"),
        ("days", "Days", "text"),
        ("any_hit_record", "Any-hit record", "text"),
        ("any_hit_rate", "Any-hit rate", "percent"),
        ("both_hit_record", "Both-hit record", "text"),
        ("both_hit_rate", "Both-hit rate", "percent"),
      ])}
      <h3>Two-Pick Card: Any Publisher Vs None</h3>
      {html_table(publisher_split.get("top2_card_has_publisher", []), [
        ("group", "Pair type", "text"),
        ("days", "Days", "text"),
        ("any_hit_record", "Any-hit record", "text"),
        ("any_hit_rate", "Any-hit rate", "percent"),
        ("both_hit_record", "Both-hit record", "text"),
        ("both_hit_rate", "Both-hit rate", "percent"),
      ])}
      <p>Read: publisher status looked mildly helpful for the single top pick, but not strong enough to be a hard rule for the two-pick card.</p>
    </section>

    <section>
      <h2>Bob Score Ranges</h2>
      {html_table(score_rows, [
        ("score_range", "Score range", "text"),
        ("decisions", "Decisions", "text"),
        ("hits", "Hits", "text"),
        ("hit_rate", "Hit rate", "percent"),
        ("pickable_rate", "Pickable", "percent"),
        ("publisher_rate", "Publisher", "percent"),
        ("avg_stop_count", "Avg stops", "float1"),
      ])}
    </section>

    <div class="two">
      <section>
        <h2>Highest-Impact Binary Factors</h2>
        {html_table(top_binary, [
          ("factor", "Factor", "text"),
          ("yes_decisions", "Yes decisions", "text"),
          ("yes_hit_rate", "Yes rate", "percent"),
          ("no_hit_rate", "No rate", "percent"),
          ("edge_vs_no", "Edge", "signed_percent"),
        ])}
      </section>
      <section>
        <h2>Strongest Numeric Cohorts</h2>
        {html_table(top_numeric, [
          ("feature", "Feature", "text"),
          ("decisions", "Decisions", "text"),
          ("correlation", "Corr", "float3"),
          ("bottom_quintile_hit_rate", "Bottom Q", "percent"),
          ("top_quintile_hit_rate", "Top Q", "percent"),
          ("top_minus_bottom", "Top edge", "signed_percent"),
        ])}
      </section>
    </div>

    <section>
      <h2>Stop Valve Outcomes</h2>
      {html_table(stop_rows[:30], [
        ("stop_reason", "Stop valve", "text"),
        ("decisions", "Decisions", "text"),
        ("hits", "Hits", "text"),
        ("hit_rate", "Hit rate", "percent"),
        ("edge_vs_overall", "Edge vs overall", "signed_percent"),
      ])}
    </section>

    <section>
      <h2>Notes For Future Iteration</h2>
      <ul>
        <li>Training rows are limited to explicit 1/0 hit results; postponed, pending, unresolved, and blank no-decision rows are excluded from rate calculations.</li>
        <li>The walk-forward model is refit for each date using only earlier labeled dates, then daily policies select from that date's candidate pool.</li>
        <li>CSV copies of these tables are written next to this report so the cohorts can be sorted or filtered later.</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""


def build_report(args: argparse.Namespace) -> dict:
    candidates_path = Path(args.candidates)
    df = prepare_dataframe(candidates_path)
    df = add_engineered_columns(df, Path(args.congregation_csv))

    overview = overview_summary(df)
    score_rows = score_bucket_summary(df)
    binary_rows = binary_factor_summary(df)
    numeric_rows = numeric_feature_summary(df)
    stop_rows = stop_reason_summary(df)
    strategy_rows, daily_rows = run_walk_forward_strategy(df, args.min_train_dates)
    best_top1_strategy = max(strategy_rows, key=lambda row: row.get("top1_hit_rate") or 0.0) if strategy_rows else {}
    best_top2_strategy = max(strategy_rows, key=lambda row: row.get("top2_any_hit_rate") or 0.0) if strategy_rows else {}
    publisher_split = publisher_split_summary(daily_rows, "wf_probability_top")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(args.report_dir)
    table_dir = report_dir / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / f"selection_strategy_report_{timestamp}.html"
    latest_html = Path(args.latest_html)
    web_html = Path(args.web_html)

    tables = {
        "score_ranges_csv": table_dir / "score_ranges.csv",
        "binary_factors_csv": table_dir / "binary_factors.csv",
        "numeric_features_csv": table_dir / "numeric_features.csv",
        "stop_reasons_csv": table_dir / "stop_reasons.csv",
        "strategy_summary_csv": table_dir / "strategy_summary.csv",
        "strategy_daily_csv": table_dir / "strategy_daily.csv",
    }
    write_csv(tables["score_ranges_csv"], score_rows)
    write_csv(tables["binary_factors_csv"], binary_rows)
    write_csv(tables["numeric_features_csv"], numeric_rows)
    write_csv(tables["stop_reasons_csv"], stop_rows)
    write_csv(tables["strategy_summary_csv"], strategy_rows)
    write_csv(tables["strategy_daily_csv"], daily_rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_candidates": str(candidates_path.resolve()),
        "min_train_dates": args.min_train_dates,
        "overview": overview,
        "best_top1_strategy": best_top1_strategy,
        "best_top2_strategy": best_top2_strategy,
        "publisher_split": publisher_split,
        "strategy_summaries": strategy_rows,
        "score_ranges": score_rows,
        "binary_factors": binary_rows,
        "numeric_features": numeric_rows,
        "stop_reasons": stop_rows,
        "report_html": str(html_path.resolve()),
        "latest_html": str(latest_html.resolve()),
        "web_report_html": str(web_html.resolve()),
        "web_report_url": "data/selection_strategy_report.html",
        "tables": {key: str(path.resolve()) for key, path in tables.items()},
    }

    html_report = render_html_report(
        payload=payload,
        score_rows=score_rows,
        binary_rows=binary_rows,
        numeric_rows=numeric_rows,
        stop_rows=stop_rows,
        strategy_rows=strategy_rows,
        publisher_split=publisher_split,
    )
    html_path.write_text(html_report, encoding="utf-8")
    latest_html.parent.mkdir(parents=True, exist_ok=True)
    latest_html.write_text(html_report, encoding="utf-8")
    web_html.parent.mkdir(parents=True, exist_ok=True)
    web_html.write_text(html_report, encoding="utf-8")
    write_json(Path(args.model_json), payload)
    write_json(Path(args.web_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Statbirt factor and daily selection strategy research report.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--congregation-csv", default=str(DEFAULT_CONGREGATION_CSV))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--latest-html", default=str(DEFAULT_WEB_REPORT))
    parser.add_argument("--web-html", default=str(DEFAULT_WEB_HTML))
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--web-json", default=str(DEFAULT_WEB_JSON))
    parser.add_argument("--min-train-dates", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_report(args)
    print(f"Selection strategy report: {payload['latest_html']}")
    best = (payload.get("strategy_summaries") or [{}])[0]
    if best:
        print(f"Best top-2 strategy: {best.get('label')} top-2 any {format_rate(best.get('top2_any_hit_rate'))}")
    best_top1 = payload.get("best_top1_strategy") or {}
    if best_top1:
        print(f"Best top-1 strategy: {best_top1.get('label')} top-1 {format_rate(best_top1.get('top1_hit_rate'))}")
    print(f"JSON: {Path(args.model_json).resolve()}")


if __name__ == "__main__":
    main()
