from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = DATA_DIR / "statbirt_candidates.csv"
DEFAULT_PREDICTIONS = DATA_DIR / "model_predictions.csv"
DEFAULT_MODEL_REPORT = DATA_DIR / "models" / "hit_probability_report.json"
DEFAULT_SHADOW_REPORT = DATA_DIR / "models" / "learned_shadow_report.json"
DEFAULT_SHADOW_PROMOTION = DATA_DIR / "models" / "learned_shadow_promotion.json"
DEFAULT_BACKTEST_DIR = PROJECT_ROOT / "logs" / "model_backtests"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "learned_model_review"
DEFAULT_LATEST_REPORT = PROJECT_ROOT / "reports" / "learned_model_review_latest.html"
DEFAULT_WEB_REPORT = PROJECT_ROOT / "web" / "data" / "learned_model_review.html"

JOIN_KEYS = ["date", "player_id", "game_pk"]
PROBABILITY_BUCKETS = [
    ("Below 55%", -math.inf, 0.55),
    ("55-60%", 0.55, 0.60),
    ("60-65%", 0.60, 0.65),
    ("65-70%", 0.65, 0.70),
    ("70-75%", 0.70, 0.75),
    ("75-80%", 0.75, 0.80),
    ("80% and above", 0.80, math.inf),
]

NUMERIC_FEATURES = [
    "learned_hit_probability",
    "score",
    "expected_pa",
    "lineup_slot",
    "hitter_pa_per_game_season",
    "hitter_last_5_games_ab",
    "hitter_last_5_games_ba",
    "hitter_ba_season",
    "hitter_ba_500_ab",
    "hitter_hipa_500_pa",
    "hitter_ba_current_hand_season",
    "hitter_pa_current_hand_season",
    "hitter_ba_current_hand_500",
    "hitter_ba_current_hand_1500",
    "hitter_k_rate_500_pa",
    "hitter_whiff_rate_500_pa",
    "pitcher_hpi_season",
    "pitcher_stuff_plus",
    "pitcher_lr_opp_ba",
    "h2h_pa",
    "h2h_xba",
    "inferred_pitch_type_xba",
    "bullpen_opp_ba",
    "park_hit_factor",
    "stop_count",
]

FEATURE_LABELS = {
    "learned_hit_probability": "Learned probability",
    "score": "Bob score",
    "expected_pa": "Expected PA",
    "lineup_slot": "Lineup slot",
    "hitter_pa_per_game_season": "Season PA per game",
    "hitter_last_5_games_ab": "Last-five AB",
    "hitter_last_5_games_ba": "Last-five BA",
    "hitter_ba_season": "Current-season batting average",
    "hitter_ba_500_ab": "500-AB batting average",
    "hitter_hipa_500_pa": "500-PA hits per PA",
    "hitter_ba_current_hand_season": "Season BA vs today's pitcher hand",
    "hitter_pa_current_hand_season": "Season PA vs today's pitcher hand",
    "hitter_ba_current_hand_500": "500-window BA vs today's pitcher hand",
    "hitter_ba_current_hand_1500": "1500-window BA vs today's pitcher hand",
    "hitter_k_rate_500_pa": "500-PA strikeout rate",
    "hitter_whiff_rate_500_pa": "500-PA whiff rate",
    "pitcher_hpi_season": "Starter season H/IP",
    "pitcher_stuff_plus": "Starter Stuff+",
    "pitcher_lr_opp_ba": "Starter BA allowed to hitter hand",
    "h2h_pa": "H2H plate appearances",
    "h2h_xba": "H2H xBA",
    "inferred_pitch_type_xba": "Pitch-type matchup xBA",
    "bullpen_opp_ba": "Bullpen opponent BA",
    "park_hit_factor": "Park hit factor",
    "stop_count": "Stop-valve count",
}


def explicit_label(value: object) -> int | None:
    text = str(value or "").strip()
    if text == "1":
        return 1
    if text == "0":
        return 0
    return None


def split_pipe(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def rate(hits: int, decisions: int) -> float | None:
    return None if decisions <= 0 else hits / decisions


def format_rate(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{value * 100:.{digits}f}%"


def format_number(value: object, digits: int = 3) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(parsed):
        return "--"
    return f"{parsed:.{digits}f}"


def wilson_interval(hits: int, decisions: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if decisions <= 0:
        return None, None
    p = hits / decisions
    denominator = 1 + z * z / decisions
    center = (p + z * z / (2 * decisions)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * decisions)) / decisions) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def longest_streak(values: Iterable[int]) -> int:
    best = 0
    current = 0
    for value in values:
        if int(value) == 1:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def prospective_mask(frame: pd.DataFrame) -> pd.Series:
    trained = pd.to_datetime(frame.get("model_trained_at", ""), errors="coerce", utc=True)
    game_start = pd.to_datetime(frame.get("game_start_time_utc", ""), errors="coerce", utc=True)
    game_date = pd.to_datetime(frame.get("date", ""), errors="coerce", utc=True)
    before_first_pitch = trained.notna() & game_start.notna() & trained.le(game_start)
    date_fallback = (
        trained.notna()
        & game_start.isna()
        & game_date.notna()
        & pd.Series(trained.dt.date, index=frame.index).le(pd.Series(game_date.dt.date, index=frame.index))
    )
    return before_first_pitch | date_fallback


def auc_score(labels: pd.Series, probabilities: pd.Series) -> float | None:
    usable = pd.DataFrame({"label": labels, "probability": probabilities}).dropna()
    positives = int(usable["label"].sum())
    negatives = int(len(usable) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = usable["probability"].rank(method="average")
    rank_sum = float(ranks[usable["label"].eq(1)].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def resolved_metrics(frame: pd.DataFrame) -> dict:
    resolved = frame[frame["label"].notna()].copy()
    decisions = int(len(resolved))
    hits = int(resolved["label"].sum())
    probabilities = pd.to_numeric(resolved.get("learned_hit_probability", np.nan), errors="coerce")
    labels = pd.to_numeric(resolved.get("label", np.nan), errors="coerce")
    usable = probabilities.notna() & labels.notna()
    if usable.any():
        clipped = probabilities[usable].clip(1e-6, 1 - 1e-6)
        y = labels[usable]
        brier = float(((clipped - y) ** 2).mean())
        log_loss = float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())
    else:
        brier = None
        log_loss = None
    low, high = wilson_interval(hits, decisions)
    return {
        "decisions": decisions,
        "hits": hits,
        "misses": decisions - hits,
        "hit_rate": rate(hits, decisions),
        "wilson_low": low,
        "wilson_high": high,
        "brier": brier,
        "log_loss": log_loss,
        "roc_auc": auc_score(labels, probabilities),
    }


def ranked_summary(frame: pd.DataFrame) -> dict:
    output: dict[str, dict] = {}
    for rank_limit in (1, 2, 5, 10):
        subset = frame[frame["learned_rank"].le(rank_limit)]
        output[f"top_{rank_limit}_individual"] = resolved_metrics(subset)

    cards = []
    for date_value, rows in frame.sort_values("learned_rank").groupby("date"):
        top_two = rows.head(2)
        labels = top_two["label"].dropna()
        if len(top_two) != 2 or len(labels) != 2:
            continue
        cards.append(
            {
                "date": str(date_value),
                "any_hit": int(labels.eq(1).any()),
                "both_hit": int(labels.eq(1).all()),
            }
        )
    card_frame = pd.DataFrame(cards)
    decisions = int(len(card_frame))
    any_hits = int(card_frame.get("any_hit", pd.Series(dtype=int)).sum())
    both_hits = int(card_frame.get("both_hit", pd.Series(dtype=int)).sum())
    output["top_2_cards"] = {
        "decisions": decisions,
        "any_hits": any_hits,
        "any_hit_rate": rate(any_hits, decisions),
        "both_hits": both_hits,
        "both_hit_rate": rate(both_hits, decisions),
        "any_hit_max_streak": longest_streak(card_frame.get("any_hit", [])),
    }

    resolved_dates = sorted(frame.loc[frame["label"].notna(), "date"].unique())
    midpoint = len(resolved_dates) // 2
    era_rows = []
    for label, dates in (("Earlier prospective dates", resolved_dates[:midpoint]), ("Recent prospective dates", resolved_dates[midpoint:])):
        top_one = frame[frame["date"].isin(dates) & frame["learned_rank"].eq(1)]
        summary = resolved_metrics(top_one)
        era_rows.append({"period": label, "dates": len(dates), **summary})
    output["top_1_by_era"] = era_rows
    return output


def calibration_summary(frame: pd.DataFrame) -> list[dict]:
    resolved = frame[frame["label"].notna()].copy()
    probability = pd.to_numeric(resolved.get("learned_hit_probability", np.nan), errors="coerce")
    output = []
    for label, low, high in PROBABILITY_BUCKETS:
        if math.isinf(low):
            mask = probability.lt(high)
        elif math.isinf(high):
            mask = probability.ge(low)
        else:
            mask = probability.ge(low) & probability.lt(high)
        rows = resolved[mask]
        decisions = int(len(rows))
        hits = int(rows["label"].sum())
        observed = rate(hits, decisions)
        average_probability = None if decisions == 0 else float(rows["learned_hit_probability"].mean())
        low_ci, high_ci = wilson_interval(hits, decisions)
        output.append(
            {
                "probability_band": label,
                "decisions": decisions,
                "hits": hits,
                "misses": decisions - hits,
                "average_probability": average_probability,
                "observed_hit_rate": observed,
                "calibration_gap": None if observed is None or average_probability is None else observed - average_probability,
                "wilson_low": low_ci,
                "wilson_high": high_ci,
            }
        )
    return output


def numeric_signal_summary(top_five: pd.DataFrame) -> list[dict]:
    resolved = top_five[top_five["label"].notna()].copy()
    output = []
    for feature in NUMERIC_FEATURES:
        if feature not in resolved.columns:
            continue
        values = pd.to_numeric(resolved[feature], errors="coerce")
        hits = values[resolved["label"].eq(1)].dropna()
        misses = values[resolved["label"].eq(0)].dropna()
        standard_deviation = float(values.std(ddof=0)) if values.notna().sum() else 0.0
        if len(hits) < 5 or len(misses) < 5 or not math.isfinite(standard_deviation) or standard_deviation < 1e-9:
            continue
        hit_mean = float(hits.mean())
        miss_mean = float(misses.mean())
        output.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "hit_mean": hit_mean,
                "miss_mean": miss_mean,
                "standardized_difference": (hit_mean - miss_mean) / standard_deviation,
                "hit_rows": int(len(hits)),
                "miss_rows": int(len(misses)),
            }
        )
    return sorted(output, key=lambda row: abs(row["standardized_difference"]), reverse=True)


def threshold_signal_summary(top_five: pd.DataFrame) -> list[dict]:
    resolved = top_five[top_five["label"].notna()].copy()
    flags = [
        ("Confirmed official lineup", resolved.get("confirmed_lineup", "").astype(str).str.upper().eq("Y")),
        ("Expected PA at least 4.2", resolved["expected_pa"].ge(4.2)),
        ("Season PA/game at least 4.2", resolved["hitter_pa_per_game_season"].ge(4.2)),
        ("Batting in top five", resolved["lineup_slot"].le(5)),
        ("500-PA K rate at most 20%", resolved["hitter_k_rate_500_pa"].le(0.20)),
        ("500-AB batting average at least .275", resolved["hitter_ba_500_ab"].ge(0.275)),
        ("Starter season H/IP at least 1.00", resolved["pitcher_hpi_season"].ge(1.00)),
        ("Starter Stuff+ at most 100", resolved["pitcher_stuff_plus"].le(100)),
        ("Four or fewer stop valves", resolved["stop_count"].le(4)),
        ("H2H xBA at least .275", resolved["h2h_xba"].ge(0.275)),
        ("Pitch-type matchup xBA at least .300", resolved["inferred_pitch_type_xba"].ge(0.300)),
    ]
    output = []
    for label, mask in flags:
        yes = resolved[mask.fillna(False)]
        no = resolved[~mask.fillna(False)]
        if len(yes) < 10 or len(no) < 10:
            continue
        yes_hits = int(yes["label"].sum())
        no_hits = int(no["label"].sum())
        yes_rate = rate(yes_hits, len(yes))
        no_rate = rate(no_hits, len(no))
        output.append(
            {
                "factor": label,
                "yes_decisions": int(len(yes)),
                "yes_hits": yes_hits,
                "yes_hit_rate": yes_rate,
                "no_decisions": int(len(no)),
                "no_hits": no_hits,
                "no_hit_rate": no_rate,
                "edge": None if yes_rate is None or no_rate is None else yes_rate - no_rate,
            }
        )
    return sorted(output, key=lambda row: abs(row.get("edge") or 0.0), reverse=True)


def stop_reason_summary(top_five: pd.DataFrame) -> list[dict]:
    resolved = top_five[top_five["label"].notna()].copy()
    reasons: dict[str, list[int]] = defaultdict(list)
    for _, row in resolved.iterrows():
        for reason in split_pipe(row.get("hard_pass_reasons", "")):
            reasons[reason].append(int(row["label"]))
    output = []
    for reason, labels in reasons.items():
        if len(labels) < 8:
            continue
        hits = sum(labels)
        output.append(
            {
                "reason": reason,
                "decisions": len(labels),
                "hits": hits,
                "misses": len(labels) - hits,
                "hit_rate": hits / len(labels),
            }
        )
    return sorted(output, key=lambda row: row["hit_rate"])


def miss_forensics(prospective: pd.DataFrame) -> list[dict]:
    misses = prospective[prospective["learned_rank"].eq(1) & prospective["label"].eq(0)].copy()
    output = []
    for _, row in misses.sort_values("date").iterrows():
        warnings = []
        if row.get("expected_pa", 9) < 4.2:
            warnings.append("low expected PA")
        if row.get("lineup_slot", 0) > 5:
            warnings.append("lineup slot 6+")
        if row.get("hitter_pa_per_game_season", 9) < 4.2:
            warnings.append("low season PA/game")
        if row.get("hitter_ba_500_ab", 9) < 0.275:
            warnings.append("500-AB BA below .275")
        if row.get("hitter_k_rate_500_pa", 0) > 0.22:
            warnings.append("500-PA K rate above 22%")
        if row.get("pitcher_stuff_plus", 0) > 100:
            warnings.append("Stuff+ above 100")
        if row.get("stop_count", 0) > 4:
            warnings.append("five or more stop valves")
        output.append(
            {
                "date": str(row.get("date", "")),
                "player": str(row.get("player_pred", row.get("player", ""))),
                "probability": row.get("learned_hit_probability"),
                "warnings": ", ".join(warnings) if warnings else "No simple pregame warning from the tested set",
                "expected_pa": row.get("expected_pa"),
                "lineup_slot": row.get("lineup_slot"),
                "ba_500": row.get("hitter_ba_500_ab"),
                "stuff_plus": row.get("pitcher_stuff_plus"),
                "stop_count": int(row.get("stop_count", 0)),
            }
        )
    return output


def shadow_policy_summary(prospective: pd.DataFrame) -> list[dict]:
    top_five = prospective[prospective["learned_rank"].le(5)].copy()
    dates = sorted(top_five["date"].unique())
    recent_dates = set(dates[len(dates) // 2 :])
    policies = [
        ("Raw learned rank", lambda rows: rows),
        ("Prefer 500-AB BA >= .275", lambda rows: rows[rows["hitter_ba_500_ab"].ge(0.275)]),
        ("Prefer starter Stuff+ <= 100", lambda rows: rows[rows["pitcher_stuff_plus"].le(100)]),
        (
            "Prefer BA >= .275 and Stuff+ <= 100",
            lambda rows: rows[rows["hitter_ba_500_ab"].ge(0.275) & rows["pitcher_stuff_plus"].le(100)],
        ),
    ]
    output = []
    for label, selector in policies:
        picks = []
        for date_value, rows in top_five.groupby("date"):
            ranked = rows.sort_values("learned_rank")
            eligible = selector(ranked)
            pick = (eligible if not eligible.empty else ranked).iloc[0]
            picks.append({"date": str(date_value), "label": pick.get("label")})
        pick_frame = pd.DataFrame(picks)
        all_resolved = pick_frame[pick_frame["label"].notna()]
        recent_resolved = pick_frame[pick_frame["date"].isin(recent_dates) & pick_frame["label"].notna()]
        all_hits = int(all_resolved["label"].sum())
        recent_hits = int(recent_resolved["label"].sum())
        output.append(
            {
                "policy": label,
                "hits": all_hits,
                "decisions": int(len(all_resolved)),
                "hit_rate": rate(all_hits, len(all_resolved)),
                "recent_hits": recent_hits,
                "recent_decisions": int(len(recent_resolved)),
                "recent_hit_rate": rate(recent_hits, len(recent_resolved)),
            }
        )
    return output


def latest_backtest(path: Path | None, directory: Path) -> tuple[dict | None, str]:
    if path is not None:
        candidates = [path]
    else:
        candidates = sorted(directory.glob("learned_model_backtest_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        return None, ""
    selected = candidates[0]
    return json.loads(selected.read_text(encoding="utf-8")), str(selected.resolve())


def prepare_frames(candidates_path: Path, predictions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(candidates_path, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
    if candidates.duplicated(JOIN_KEYS).any() or predictions.duplicated(JOIN_KEYS).any():
        raise ValueError("Candidate and prediction rows must be unique by date/player_id/game_pk.")

    merged = predictions.merge(
        candidates,
        on=JOIN_KEYS,
        how="left",
        suffixes=("_pred", ""),
        validate="one_to_one",
        indicator=True,
    )
    merged["label"] = merged.get("result_hit", "").map(explicit_label)
    merged["learned_rank"] = pd.to_numeric(merged.get("learned_rank", np.nan), errors="coerce")
    merged["learned_hit_probability"] = pd.to_numeric(
        merged.get("learned_hit_probability", np.nan), errors="coerce"
    )
    merged["stop_count"] = merged.get("hard_pass_reasons", "").map(lambda value: len(split_pipe(value)))
    hand = merged.get("pitcher_hand", "").astype(str).str.upper()
    merged["hitter_ba_current_hand_season"] = np.where(
        hand.eq("L"), merged.get("hitter_split_ba_season_vs_lhp", ""), merged.get("hitter_split_ba_season_vs_rhp", "")
    )
    merged["hitter_pa_current_hand_season"] = np.where(
        hand.eq("L"), merged.get("hitter_split_pa_season_vs_lhp", ""), merged.get("hitter_split_pa_season_vs_rhp", "")
    )
    merged["hitter_ba_current_hand_500"] = np.where(
        hand.eq("L"), merged.get("hitter_split_ba_500_vs_lhp", ""), merged.get("hitter_split_ba_500_vs_rhp", "")
    )
    merged["hitter_ba_current_hand_1500"] = np.where(
        hand.eq("L"), merged.get("hitter_split_ba_1500_vs_lhp", ""), merged.get("hitter_split_ba_1500_vs_rhp", "")
    )
    for feature in set(NUMERIC_FEATURES):
        if feature in merged.columns:
            merged[feature] = pd.to_numeric(merged[feature], errors="coerce")
    merged["candidate_match"] = merged["_merge"].eq("both")
    invalid_rank_dates = set()
    for date_value, rows in merged[merged["candidate_match"]].groupby("date"):
        ranks = rows["learned_rank"]
        if ranks.isna().any() or ranks.duplicated().any() or ranks.min() != 1 or ranks.nunique() != len(rows):
            invalid_rank_dates.add(str(date_value))
    merged["ranking_valid"] = ~merged["date"].isin(invalid_rank_dates)
    merged["prospective_raw"] = prospective_mask(merged)
    merged["prospective"] = merged["prospective_raw"] & merged["candidate_match"] & merged["ranking_valid"]
    return candidates, merged


def html_table(rows: list[dict], columns: list[tuple[str, str, str]], *, empty: str = "No data available.") -> str:
    if not rows:
        return f'<p class="muted">{html.escape(empty)}</p>'
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = []
    for row in rows:
        cells = []
        for key, _, kind in columns:
            value = row.get(key)
            if kind == "percent":
                display = format_rate(value)
            elif kind == "signed_percent":
                display = "--" if value is None else f"{float(value) * 100:+.1f} pp"
            elif kind == "decimal2":
                display = format_number(value, 2)
            elif kind == "decimal3":
                display = format_number(value, 3)
            else:
                display = "--" if value is None or value == "" else str(value)
            cells.append(f"<td>{html.escape(display)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def metric_card(label: str, value: str, note: str) -> str:
    return (
        '<article class="metric">'
        f"<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small>"
        "</article>"
    )


def render_report(payload: dict) -> str:
    prospective = payload["prospective"]
    ranks = prospective["ranked_summary"]
    top_one = ranks["top_1_individual"]
    top_two = ranks["top_2_cards"]
    recent_era = ranks["top_1_by_era"][-1] if ranks["top_1_by_era"] else {}
    validation = payload.get("production_model", {}).get("validation_metrics") or {}
    top_five = ranks["top_5_individual"]
    backtests = (payload.get("walk_forward_backtest") or {}).get("summaries") or []
    backtest_rows = [
        {
            "name": row.get("name", ""),
            "description": row.get("description", ""),
            "evaluated_dates": row.get("evaluated_dates"),
            "record": f"{row.get('wins', 0)}-{row.get('losses', 0)}",
            "daily_success_rate": row.get("daily_success_rate"),
            "max_success_streak": row.get("max_success_streak"),
        }
        for row in backtests
    ]

    high_band = next(
        (row for row in prospective["calibration"] if row["probability_band"] == "75-80%"),
        {},
    )
    strongest_thresholds = prospective["threshold_signals"][:6]
    raw_shadow = next((row for row in prospective["shadow_policies"] if row["policy"] == "Raw learned rank"), {})
    stuff_shadow = next(
        (row for row in prospective["shadow_policies"] if row["policy"] == "Prefer starter Stuff+ <= 100"),
        {},
    )
    miss_warning_count = sum(
        1 for row in prospective["rank_one_misses"] if not row["warnings"].startswith("No simple")
    )
    report_date = payload["report_date"]
    shadow_model = payload.get("shadow_model") or {}
    shadow_validation = shadow_model.get("untouched_ranking_validation") or {}
    shadow_promotion = payload.get("shadow_promotion") or {}

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Statbirt Learned Model Review - {html.escape(report_date)}</title>
    <style>
      :root {{ color-scheme: light; --ink:#17212b; --muted:#5d6975; --line:#d8dee5; --soft:#f4f6f8; --green:#17643d; --green-bg:#e8f4ed; --amber:#855a00; --amber-bg:#fff3d6; --red:#9a302c; --red-bg:#fae9e7; --blue:#245a86; }}
      * {{ box-sizing:border-box; }}
      body {{ margin:0; background:#eef1f4; color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }}
      main {{ width:min(1180px,calc(100% - 32px)); margin:24px auto 48px; background:white; border:1px solid var(--line); }}
      header {{ padding:30px 34px 26px; border-bottom:1px solid var(--line); }}
      .eyebrow {{ margin:0 0 6px; color:var(--blue); font-size:12px; font-weight:800; text-transform:uppercase; }}
      h1 {{ margin:0; font-size:clamp(28px,4vw,44px); line-height:1.08; }}
      header p {{ max-width:850px; margin:12px 0 0; color:var(--muted); }}
      nav {{ display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:18px; }}
      a {{ color:var(--blue); }}
      section {{ padding:26px 34px; border-bottom:1px solid var(--line); }}
      h2 {{ margin:0 0 8px; font-size:23px; }}
      h3 {{ margin:24px 0 8px; font-size:17px; }}
      p {{ margin:8px 0 12px; }}
      .lede {{ font-size:17px; max-width:920px; }}
      .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:18px; }}
      .metric {{ min-height:124px; padding:16px; border:1px solid var(--line); border-top:4px solid var(--blue); }}
      .metric span,.metric small {{ display:block; color:var(--muted); }}
      .metric strong {{ display:block; margin:4px 0; font-size:28px; line-height:1.1; }}
      .callout {{ margin:16px 0; padding:15px 17px; border-left:5px solid var(--amber); background:var(--amber-bg); }}
      .good {{ border-left-color:var(--green); background:var(--green-bg); }}
      .risk {{ border-left-color:var(--red); background:var(--red-bg); }}
      .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
      .panel {{ padding:18px; border:1px solid var(--line); }}
      .panel h3 {{ margin-top:0; }}
      ul,ol {{ padding-left:22px; }}
      li {{ margin:7px 0; }}
      .table-wrap {{ width:100%; overflow-x:auto; border:1px solid var(--line); margin:12px 0 20px; }}
      table {{ width:100%; min-width:720px; border-collapse:collapse; font-size:13px; }}
      th,td {{ padding:9px 10px; text-align:left; vertical-align:top; border-bottom:1px solid var(--line); }}
      th {{ background:var(--soft); font-size:12px; white-space:nowrap; }}
      tr:last-child td {{ border-bottom:0; }}
      .muted,small {{ color:var(--muted); }}
      footer {{ padding:20px 34px; color:var(--muted); font-size:13px; overflow-wrap:anywhere; word-break:break-word; }}
      code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
      @media (max-width:850px) {{ .metrics,.grid {{ grid-template-columns:1fr 1fr; }} }}
      @media (max-width:560px) {{ main {{ width:100%; margin:0; border-left:0; border-right:0; }} header,section,footer {{ padding-left:18px; padding-right:18px; }} .metrics,.grid {{ grid-template-columns:1fr; }} .metric {{ min-height:0; }} }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <p class="eyebrow">Statbirt research archive</p>
        <h1>Learned Model Review</h1>
        <p>{html.escape(report_date)} | Candidate history through {html.escape(payload['data_date_max'])} | A performance, calibration, and feature-weight audit focused on choosing one or two hitters per day.</p>
        <nav><a href="../index.html">Bob score board</a><a href="../learned.html">Learned shortlist</a><a href="selection_strategy_report.html">6-19-26 Strategy Report</a></nav>
      </header>

      <section>
        <h2>Executive verdict</h2>
        <p class="lede"><strong>The learned model is useful as a shortlist generator, but its upper probability values and rank order should not currently be treated as literal confidence.</strong> The clean prospective record is materially lower than the backfilled historical headline, and recent results show that stable hitter quality and opposing Stuff+ are doing more work inside the top five than small differences in learned probability.</p>
        <div class="metrics">
          {metric_card('Prospective rank-one', f"{top_one['hits']}/{top_one['decisions']}", format_rate(top_one['hit_rate']))}
          {metric_card('Prospective top-two any hit', f"{top_two['any_hits']}/{top_two['decisions']}", format_rate(top_two['any_hit_rate']))}
          {metric_card('Prospective top-five individuals', f"{top_five['hits']}/{top_five['decisions']}", format_rate(top_five['hit_rate']))}
          {metric_card('Recent-half rank-one', f"{recent_era.get('hits', 0)}/{recent_era.get('decisions', 0)}", format_rate(recent_era.get('hit_rate')))}
        </div>
        <div class="callout risk"><strong>Do not quote the old 97.7% as expected future accuracy.</strong> It came from chronological model fits on historical boards often constructed with final boxscore batting orders and some full-season context unavailable on the original date. The clean, logged pregame top-two record is {top_two['any_hits']}/{top_two['decisions']} ({format_rate(top_two['any_hit_rate'])}).</div>
        <div class="callout good"><strong>What remains encouraging:</strong> the prospective top five have produced {top_five['hits']} hits in {top_five['decisions']} resolved selections ({format_rate(top_five['hit_rate'])}), and the top-two card has still found at least one hit on {top_two['any_hits']} of {top_two['decisions']} fully resolved days.</div>
      </section>

      <section>
        <h2>Evidence hierarchy</h2>
        <div class="grid">
          <article class="panel"><h3>Use for decisions</h3><ul><li>Predictions whose model training timestamp is before first pitch.</li><li>Chronological walk-forward comparisons.</li><li>Resolved appearances only; postponed and no-appearance rows remain ungraded.</li><li>Confidence intervals and recent-period splits.</li></ul></article>
          <article class="panel"><h3>Use only for hypothesis generation</h3><ul><li>Historical predictions rescored by a model trained after those games.</li><li>Backfilled rows whose final batting order was used as a lineup proxy.</li><li>Backfilled full-season pitcher, park, or sprint context that was not bounded to the game date.</li><li>Small probability bands, mined cohorts, or Publisher status applied retrospectively.</li></ul></article>
        </div>
        <div class="callout risk"><strong>Point-in-time limitation:</strong> 2025 backfills use the final boxscore lineup by default. The current pipeline also obtains some pitcher season context without a historical date bound, while season park factors and sprint speed can contain later-season information. These fields must be rebuilt point-in-time before historical rates can be called fully leakage-safe.</div>
        <div class="callout"><strong>Target limitation:</strong> {payload['data_quality']['candidate_no_appearance_rows']:,} no-appearance rows are intentionally ungraded and excluded from training. The saved probability therefore behaves more like P(hit | the hitter appears) than the morning decision's P(hit). A two-stage appearance model, multiplied by hit-given-appearance probability, would align training with the actual pre-lineup choice.</div>
        <h3>Production holdout metrics</h3>
        <p>The current model's date-based 80/20 validation slice contains {validation.get('rows', '--')} labeled rows. These metrics describe broad discrimination among players later known to have appeared, not the full morning candidate board or the daily single-pick problem.</p>
        {html_table([{
            'rows': validation.get('rows'), 'hit_rate': validation.get('hit_rate'), 'roc_auc': validation.get('roc_auc'),
            'brier': validation.get('brier'), 'log_loss': validation.get('log_loss'), 'top_10_hit_rate': validation.get('top_10_hit_rate')
        }], [('rows','Rows','text'),('hit_rate','Base hit rate','percent'),('roc_auc','ROC AUC','decimal3'),('brier','Brier','decimal3'),('log_loss','Log loss','decimal3'),('top_10_hit_rate','Top-10 hit rate','percent')])}
      </section>

      <section>
        <h2>Prediction archive health</h2>
        <p>The saved prediction CSV is useful but is not yet an immutable decision ledger. Its upsert replaces matching player/game keys without deleting obsolete rows from the same date.</p>
        {html_table([
            {'check':'Prediction rows with no current candidate match','value':payload['data_quality']['orphan_prediction_rows'],'meaning':'Stale or superseded prediction keys'},
            {'check':'Dates excluded for invalid or duplicate ranks','value':len(payload['data_quality']['invalid_rank_dates']),'meaning':', '.join(payload['data_quality']['invalid_rank_dates']) or 'None'},
            {'check':'Prediction results lagging candidate results','value':payload['data_quality']['prediction_result_lag_rows'],'meaning':'Candidate CSV used as the authoritative label'},
            {'check':'Candidate no-appearance rows','value':payload['data_quality']['candidate_no_appearance_rows'],'meaning':'Kept ungraded, per project policy'},
        ], [('check','Check','text'),('value','Count','text'),('meaning','Treatment','text')])}
        <div class="callout"><strong>Infrastructure recommendation:</strong> write one immutable pregame snapshot per run with a cutoff timestamp, candidate-pool source, lineup provenance, model version, ranks, and probabilities. Join results into a separate ledger instead of mutating the prediction rows.</div>
      </section>

      <section>
        <h2>Prospective performance</h2>
        <p>This is the most important table in the report. A prediction qualifies only when its saved model timestamp precedes that game's first pitch (or, where first-pitch time is unavailable, is no later than the game date).</p>
        {html_table([
            {'population':'Rank 1','decisions':ranks['top_1_individual']['decisions'],'hits':ranks['top_1_individual']['hits'],'hit_rate':ranks['top_1_individual']['hit_rate'],'interval':f"{format_rate(ranks['top_1_individual']['wilson_low'])} to {format_rate(ranks['top_1_individual']['wilson_high'])}"},
            {'population':'Ranks 1-2, individual','decisions':ranks['top_2_individual']['decisions'],'hits':ranks['top_2_individual']['hits'],'hit_rate':ranks['top_2_individual']['hit_rate'],'interval':f"{format_rate(ranks['top_2_individual']['wilson_low'])} to {format_rate(ranks['top_2_individual']['wilson_high'])}"},
            {'population':'Ranks 1-5, individual','decisions':ranks['top_5_individual']['decisions'],'hits':ranks['top_5_individual']['hits'],'hit_rate':ranks['top_5_individual']['hit_rate'],'interval':f"{format_rate(ranks['top_5_individual']['wilson_low'])} to {format_rate(ranks['top_5_individual']['wilson_high'])}"},
            {'population':'Ranks 1-10, individual','decisions':ranks['top_10_individual']['decisions'],'hits':ranks['top_10_individual']['hits'],'hit_rate':ranks['top_10_individual']['hit_rate'],'interval':f"{format_rate(ranks['top_10_individual']['wilson_low'])} to {format_rate(ranks['top_10_individual']['wilson_high'])}"},
        ], [('population','Population','text'),('hits','Hits','text'),('decisions','Decisions','text'),('hit_rate','Hit rate','percent'),('interval','Wilson 95% interval','text')])}
        <h3>Rank-one drift check</h3>
        {html_table(ranks['top_1_by_era'], [('period','Period','text'),('dates','Resolved dates','text'),('hits','Hits','text'),('decisions','Decisions','text'),('hit_rate','Hit rate','percent')])}
      </section>

      <section>
        <h2>Calibration</h2>
        <p>Probabilities are reasonably ordered through 70-75%, then become overconfident. In the 75-80% band, the prospective record is {high_band.get('hits', 0)}/{high_band.get('decisions', 0)} ({format_rate(high_band.get('observed_hit_rate'))}); at 80% and above the sample is smaller still. The correct response is calibration and reranking, not another hard stop valve.</p>
        {html_table(prospective['calibration'], [('probability_band','Predicted band','text'),('hits','Hits','text'),('decisions','Decisions','text'),('average_probability','Average prediction','percent'),('observed_hit_rate','Observed hit rate','percent'),('calibration_gap','Observed minus predicted','signed_percent')])}
      </section>

      <section>
        <h2>What separated top-five hits from misses</h2>
        <p>These are prospective associations within the learned top five. They answer the useful question, "Once the model has found five plausible hitters, what helps order them?" They do not prove causality.</p>
        <h3>Largest numeric separations</h3>
        {html_table(prospective['numeric_signals'][:12], [('label','Feature','text'),('hit_mean','Hit mean','decimal3'),('miss_mean','Miss mean','decimal3'),('standardized_difference','Standardized difference','decimal2'),('hit_rows','Hit rows','text'),('miss_rows','Miss rows','text')])}
        <h3>Readable threshold checks</h3>
        {html_table(strongest_thresholds, [('factor','Factor','text'),('yes_hits','Yes hits','text'),('yes_decisions','Yes decisions','text'),('yes_hit_rate','Yes hit rate','percent'),('no_hit_rate','No hit rate','percent'),('edge','Edge','signed_percent')])}
        <div class="callout"><strong>Current signal:</strong> 500-AB batting average and opposing Stuff+ are stronger separators than the small probability differences inside the top five. The learned probability itself is slightly higher on misses than hits in this prospective slice, which is the clearest evidence that a second-stage shortlist ranker deserves a shadow test.</div>
      </section>

      <section>
        <h2>Could the rank-one misses have been foreseen?</h2>
        <p>{miss_warning_count} of {len(prospective['rank_one_misses'])} prospective rank-one misses carried at least one simple warning available before the game. This does not mean those picks were objectively wrong; even a true 75% event fails one time in four. It does show which warnings should be presented explicitly to the daily committee.</p>
        {html_table(prospective['rank_one_misses'], [('date','Date','text'),('player','Player','text'),('probability','Probability','percent'),('warnings','Pregame warnings','text'),('expected_pa','Expected PA','decimal2'),('lineup_slot','Lineup','decimal2'),('ba_500','500-AB BA','decimal3'),('stuff_plus','Stuff+','decimal2'),('stop_count','Stops','text')])}
        <h3>Stop-valve context inside the top five</h3>
        <p>Most individual stop valves are weak filters once a hitter has already reached the learned top five. Treat them as descriptive risk flags. The broad model should learn their continuous inputs rather than accumulate binary vetoes.</p>
        {html_table(prospective['stop_reasons'], [('reason','Stop valve','text'),('hits','Hits','text'),('decisions','Decisions','text'),('hit_rate','Hit rate','percent')])}
      </section>

      <section>
        <h2>Prospective shadow rerankers</h2>
        <p>These transparent rules keep the learned top five, then prefer the highest raw rank meeting a conventional quality floor. They are post-hoc hypotheses on a small sample, not production-ready policies. The recent-half column is included to expose whether a result is driven only by the early period.</p>
        {html_table(prospective['shadow_policies'], [('policy','Policy','text'),('hits','Hits','text'),('decisions','Decisions','text'),('hit_rate','All prospective','percent'),('recent_hits','Recent hits','text'),('recent_decisions','Recent decisions','text'),('recent_hit_rate','Recent hit rate','percent')])}
        <div class="callout good"><strong>Best forward hypothesis:</strong> within the learned top five, prefer a hitter facing starter Stuff+ at or below 100 when one is available. It improved this sample from {raw_shadow.get('hits', 0)}/{raw_shadow.get('decisions', 0)} to {stuff_shadow.get('hits', 0)}/{stuff_shadow.get('decisions', 0)} overall and from {raw_shadow.get('recent_hits', 0)}/{raw_shadow.get('recent_decisions', 0)} to {stuff_shadow.get('recent_hits', 0)}/{stuff_shadow.get('recent_decisions', 0)} in the recent half. Freeze that rule now and shadow-test it prospectively; do not tune the threshold again until the evaluation window closes.</div>
      </section>

      <section>
        <h2>Walk-forward model experiments</h2>
        <p>Each model is trained only on earlier dates and asked to select the next day's rank-one hitter. The evaluator now excludes an ungraded top pick without promoting the next hitter later known to have appeared. Historical lineup and full-season context still make absolute rates optimistic, but relative comparisons are useful because every variant sees the same boards.</p>
        {html_table(backtest_rows, [('name','Experiment','text'),('description','Description','text'),('record','Record','text'),('daily_success_rate','Success rate','percent'),('max_success_streak','Longest streak','text')], empty='The walk-forward experiment output was not available when this report was built.')}
        <div class="callout good"><strong>Interpretation:</strong> the trimmed opportunity/contact model has been the strongest repeatable historical variant. Hand-tuned indicator rankers have not beaten logistic learning, and heavy stop-valve/category versions have tended to lose accuracy. Keep the continuous baseball variables; reduce brittle rules.</div>
      </section>

      <section>
        <h2>Weighting diagnosis</h2>
        <div class="grid">
          <article class="panel"><h3>Likely overweight or unstable</h3><ul><li><strong>Upper raw probability:</strong> 75%+ predictions are not calibrated prospectively.</li><li><strong>Calendar and missingness proxies:</strong> month, day of week, rain, and missing park values should not influence a pick without a baseball mechanism.</li><li><strong>Correlated short-form fields:</strong> last-five games, AB, hits, and BA create unstable signs. Replace or supplement them with AB per game and contact-quality windows.</li><li><strong>Repeated opportunity scoring:</strong> PA/game, expected PA, lineup slot, Bob score, safety, discrete bonuses, and rank penalties overlap across model and selection policy.</li><li><strong>Bob score inside the learned model:</strong> useful, but it duplicates many underlying features and can preserve old hand weights.</li><li><strong>Surface H2H:</strong> tiny samples should be shrunk strongly toward pitch-type and platoon expectations.</li></ul></article>
          <article class="panel"><h3>Likely underweighted</h3><ul><li><strong>Stable hitter contact skill:</strong> 500-AB BA and 500-PA K/whiff rates separate current top-five hits from misses.</li><li><strong>Current-season batting average:</strong> it now exists in the candidate table and separates prospective hits, but is absent from the production feature list. Use PA-shrunk season BA rather than a raw early-season rate.</li><li><strong>Starter quality at the shortlist stage:</strong> Stuff+ is more informative prospectively than its current coefficient rank suggests.</li><li><strong>Current-hand interactions:</strong> the model includes both LHP and RHP 500/1500 split values but neither current-season splits nor an explicit interaction selecting today's relevant hand.</li><li><strong>Appearance probability:</strong> lineup certainty and scratch/rest risk need a separate target before official lineups.</li><li><strong>Recency drift:</strong> the recent-half top-one record trails the earlier half, arguing for season or exponential time weights.</li><li><strong>Calibration:</strong> probabilities should be calibrated on held-out recent dates before display.</li></ul></article>
        </div>
      </section>

      <section>
        <h2>New free data worth adding</h2>
        <ol>
          <li><strong>Rolling Statcast contact quality:</strong> hitter xBA, hard-hit rate, sweet-spot rate, EV50, and the same contact allowed by the starter. This is the cleanest upgrade from outcome batting average to quality of contact.</li>
          <li><strong>Bat-tracking quality:</strong> squared-up rate, blasts, competitive-swing contact, bat speed, and swing length. Squared-up contact is especially aligned with a one-hit target.</li>
          <li><strong>Pitch-arsenal interaction:</strong> weight the hitter's xBA/contact rate against each pitch type by the probable starter's current usage, velocity, movement, and handedness, rather than using one aggregate inferred matchup.</li>
          <li><strong>Bullpen availability:</strong> pitches thrown and leverage appearances over the prior three days, expected handedness mix, and team off-days. Season bullpen BA does not tell us which relievers are available tonight.</li>
          <li><strong>Defense behind the pitcher:</strong> team and position-specific Outs Above Average. A single is partly a contact-quality event and partly whether the defense converts the ball.</li>
          <li><strong>Lineup certainty and availability:</strong> official-lineup timestamp, injury return, rest-day risk, and role changes. Historical final lineups should remain a separate dataset flag, never silently mixed with morning projections.</li>
        </ol>
        <p class="muted">Primary free references: <a href="https://www.mlb.com/glossary/statcast/expected-batting-average">MLB xBA glossary</a>, <a href="https://baseballsavant.mlb.com/leaderboard/bat-tracking">Baseball Savant bat tracking</a>, <a href="https://baseballsavant.mlb.com/expected_statistics">Baseball Savant expected statistics</a>, and <a href="https://www.mlb.com/glossary/statcast/outs-above-average">MLB Outs Above Average glossary</a>.</p>
      </section>

      <section>
        <h2>Recommended next model cycle</h2>
        <ol>
          <li>Freeze the current production learned model as the baseline; do not overwrite its historical predictions.</li>
          <li>Add a <strong>data provenance field</strong> for official, projected, recent-usage, or final-boxscore lineup source.</li>
          <li>Model <strong>P(appearance)</strong> separately, then multiply it by P(hit | appearance) for morning rankings.</li>
          <li>Build a shadow top-five reranker using stable hitter quality, Stuff+, current-hand interaction, lineup opportunity, and a small stop-count penalty. Train only on earlier dates.</li>
          <li>Remove calendar fields and raw rain from the candidate model; retain weather only through game/park context where it has a plausible mechanism.</li>
          <li>Fit a recent-date probability calibrator and show calibrated probability beside, not in place of, the raw score during evaluation.</li>
          <li>Require at least 50 prospective resolved days before promotion, with predeclared metrics: rank-one hit rate, top-two any-hit rate, longest miss streak, Brier score, and performance by season/month.</li>
        </ol>
        <div class="callout"><strong>Decision today:</strong> keep using the learned top five and the daily committee. For the single pick, treat raw ranks one and two as candidates, then explicitly compare 500-AB contact skill, K/whiff rate, lineup opportunity, opposing Stuff+, availability, and current news. Do not automatically choose the higher displayed probability when the gap is small.</div>
      </section>

      <section>
        <h2>Implementation status</h2>
        <p class="lede">The July 16 recommendations are now implemented as a production-preserving shadow experiment. The frozen learned model remains the visible ranking baseline; the shadow probability and rerank are evaluation-only until the predeclared gate is satisfied.</p>
        <div class="metrics">
          <div class="metric"><span>Shadow diagnostic top one</span><strong>{shadow_validation.get('top_one_hits', 0)}/{shadow_validation.get('resolved_top_one_days', 0)}</strong><small>{format_rate(shadow_validation.get('top_one_hit_rate'))} on untouched historical dates</small></div>
          <div class="metric"><span>Production comparison</span><strong>{shadow_validation.get('production_top_one_hits', 0)}/{shadow_validation.get('resolved_top_one_days', 0)}</strong><small>{format_rate(shadow_validation.get('production_top_one_hit_rate'))} on the same dates</small></div>
          <div class="metric"><span>Shadow top-two any hit</span><strong>{shadow_validation.get('top_two_any_hit_days', 0)}/{shadow_validation.get('resolved_top_two_days', 0)}</strong><small>{format_rate(shadow_validation.get('top_two_any_hit_rate'))} diagnostic only</small></div>
          <div class="metric"><span>Prospective gate</span><strong>{shadow_promotion.get('resolved_days', 0)}/{shadow_promotion.get('minimum_resolved_days', 50)}</strong><small>{html.escape(str(shadow_promotion.get('status') or 'collecting'))}</small></div>
        </div>
        <div class="grid">
          <article class="panel"><h3>Model and evidence controls</h3><ul><li>Frozen production artifact and versioned shadow policy.</li><li>Separate appearance and hit-given-appearance models.</li><li>Three chronological evaluation blocks plus a rolling 14-resolved-date deployment calibration holdout.</li><li>Immutable pregame snapshots with independent result joins and timestamp audits.</li><li>No automatic promotion; human review begins after 50 fully resolved prospective days.</li></ul></article>
          <article class="panel"><h3>New prospective features</h3><ul><li>Rolling xBA, hard-hit, sweet-spot, EV50, bat speed, and swing length.</li><li>Season bat-tracking squared-up, blast, and competitive-contact rates.</li><li>Starter arsenal usage with velocity and movement similarity.</li><li>Three-day bullpen workload and recent reliever usage.</li><li>Team infield and outfield Outs Above Average.</li><li>Dated roster status, activation history, and lineup provenance.</li></ul></article>
        </div>
        <div class="callout risk"><strong>Interpretation:</strong> the shadow diagnostic reranks saved production top fives, but it is still retrospective and includes legacy feature limitations. It is supporting evidence, not a promotion result. The July 18 onward immutable ledger is the deciding test.</div>
      </section>

      <footer>Generated by <code>py -3 -m statbirt.learned_review</code>. Source rows: {payload['candidate_rows']:,} candidates, {payload['prediction_rows']:,} saved predictions, {prospective['resolved_rows']:,} resolved prospective predictions. Backtest source: {html.escape(payload.get('backtest_path') or 'not available')}.</footer>
    </main>
  </body>
</html>"""


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_report(args: argparse.Namespace) -> dict:
    candidates, merged = prepare_frames(Path(args.candidates), Path(args.predictions))
    prospective = merged[merged["prospective"]].copy()
    top_five = prospective[prospective["learned_rank"].le(5)].copy()
    model_report_path = Path(args.model_report)
    model_report = json.loads(model_report_path.read_text(encoding="utf-8")) if model_report_path.exists() else {}
    shadow_report_path = Path(args.shadow_report)
    shadow_report = json.loads(shadow_report_path.read_text(encoding="utf-8")) if shadow_report_path.exists() else {}
    shadow_promotion_path = Path(args.shadow_promotion)
    shadow_promotion = (
        json.loads(shadow_promotion_path.read_text(encoding="utf-8"))
        if shadow_promotion_path.exists()
        else {}
    )
    backtest, backtest_path = latest_backtest(Path(args.backtest_json) if args.backtest_json else None, Path(args.backtest_dir))
    report_date = datetime.now().astimezone().date().isoformat()
    output_dir = Path(args.report_dir)
    tables_dir = output_dir / "tables"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    ranked = ranked_summary(prospective)
    calibration = calibration_summary(prospective)
    numeric_signals = numeric_signal_summary(top_five)
    threshold_signals = threshold_signal_summary(top_five)
    stop_reasons = stop_reason_summary(top_five)
    rank_one_misses = miss_forensics(prospective)
    shadow_policies = shadow_policy_summary(prospective)
    prediction_labels = merged.get("result_hit_pred", pd.Series("", index=merged.index)).map(explicit_label)
    candidate_labels = merged.get("result_hit", pd.Series("", index=merged.index)).map(explicit_label)
    prediction_result_lag = candidate_labels.notna() & prediction_labels.ne(candidate_labels)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_date": report_date,
        "source_candidates": str(Path(args.candidates).resolve()),
        "source_predictions": str(Path(args.predictions).resolve()),
        "candidate_rows": int(len(candidates)),
        "prediction_rows": int(len(merged)),
        "data_date_min": str(candidates["date"].min()),
        "data_date_max": str(candidates["date"].max()),
        "production_model": model_report,
        "shadow_model": shadow_report,
        "shadow_promotion": shadow_promotion,
        "backtest_path": backtest_path,
        "walk_forward_backtest": backtest,
        "data_quality": {
            "orphan_prediction_rows": int((~merged["candidate_match"]).sum()),
            "invalid_rank_dates": sorted(merged.loc[~merged["ranking_valid"], "date"].unique().tolist()),
            "prediction_result_lag_rows": int(prediction_result_lag.sum()),
            "candidate_no_appearance_rows": int(candidates.get("result_status", "").eq("no_appearance").sum()),
        },
        "prospective": {
            "rows": int(len(prospective)),
            "resolved_rows": int(prospective["label"].notna().sum()),
            "date_min": str(prospective["date"].min()) if len(prospective) else "",
            "date_max": str(prospective["date"].max()) if len(prospective) else "",
            "overall_metrics": resolved_metrics(prospective),
            "ranked_summary": ranked,
            "calibration": calibration,
            "numeric_signals": numeric_signals,
            "threshold_signals": threshold_signals,
            "stop_reasons": stop_reasons,
            "rank_one_misses": rank_one_misses,
            "shadow_policies": shadow_policies,
        },
    }

    report_html = render_report(payload)
    dated_report = output_dir / f"learned_model_review_{timestamp}.html"
    latest_report = Path(args.latest_report)
    web_report = Path(args.web_report)
    for path in (dated_report, latest_report, web_report):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_html, encoding="utf-8")

    payload["report_html"] = str(dated_report.resolve())
    payload["latest_html"] = str(latest_report.resolve())
    payload["web_report_html"] = str(web_report.resolve())
    write_json(output_dir / "learned_model_review_latest.json", payload)
    write_csv(tables_dir / "prospective_calibration.csv", calibration)
    write_csv(tables_dir / "prospective_numeric_signals.csv", numeric_signals)
    write_csv(tables_dir / "prospective_threshold_signals.csv", threshold_signals)
    write_csv(tables_dir / "prospective_rank_one_misses.csv", rank_one_misses)
    write_csv(tables_dir / "prospective_stop_reasons.csv", stop_reasons)
    write_csv(tables_dir / "prospective_shadow_policies.csv", shadow_policies)
    if backtest:
        write_csv(tables_dir / "walk_forward_experiments.csv", backtest.get("summaries") or [])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a durable learned-model performance review.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--model-report", default=str(DEFAULT_MODEL_REPORT))
    parser.add_argument("--shadow-report", default=str(DEFAULT_SHADOW_REPORT))
    parser.add_argument("--shadow-promotion", default=str(DEFAULT_SHADOW_PROMOTION))
    parser.add_argument("--backtest-dir", default=str(DEFAULT_BACKTEST_DIR))
    parser.add_argument("--backtest-json", default="")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--latest-report", default=str(DEFAULT_LATEST_REPORT))
    parser.add_argument("--web-report", default=str(DEFAULT_WEB_REPORT))
    return parser.parse_args()


def main() -> None:
    payload = build_report(parse_args())
    top_one = payload["prospective"]["ranked_summary"]["top_1_individual"]
    top_two = payload["prospective"]["ranked_summary"]["top_2_cards"]
    print(
        f"Prospective rank one: {top_one['hits']}/{top_one['decisions']} "
        f"({format_rate(top_one['hit_rate'])})"
    )
    print(
        f"Prospective top-two any hit: {top_two['any_hits']}/{top_two['decisions']} "
        f"({format_rate(top_two['any_hit_rate'])})"
    )
    print(f"Report: {payload['web_report_html']}")


if __name__ == "__main__":
    main()
