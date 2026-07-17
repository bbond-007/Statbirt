import csv
from datetime import date, timedelta
import json

from statbirt.learned_shadow import (
    APPEARANCE_FEATURES,
    HIT_FEATURES,
    appearance_label,
    decision_outcome,
    derived_features,
    promotion_report,
    score_shadow_candidates,
    train_shadow_model,
)


def row(index: int, *, day: date, status: str = "final", hit: int = 1) -> dict[str, str]:
    return {
        "date": day.isoformat(),
        "player": f"Player {index}",
        "player_id": str(1000 + index),
        "team": "AAA",
        "opponent": "BBB",
        "game_pk": str(900000 + index),
        "game_start_time_utc": f"{day.isoformat()}T23:00:00Z",
        "result_status": status,
        "result_hit": str(hit) if status == "final" else "",
        "confirmed_lineup": "Y",
        "lineup_source": "official" if index % 3 == 0 else "recent_usage",
        "candidate_pool_source": "official_lineup" if index % 3 == 0 else "recent_usage",
        "lineup_slot": str(1 + index % 8),
        "expected_pa": str(4.9 - (index % 8) * 0.12),
        "starts_last_5": str(3 + index % 3),
        "hitter_pa_per_game_season": str(3.8 + (index % 7) * 0.1),
        "hitter_last_5_games_played": "5",
        "hitter_last_5_games_ab": str(15 + index % 8),
        "hitter_ba_season": str(0.235 + (index % 8) * 0.01),
        "hitter_ba_500_ab": str(0.245 + (index % 7) * 0.008),
        "hitter_ba_2500_ab": str(0.250 + (index % 5) * 0.006),
        "hitter_hipa_500_pa": str(0.225 + (index % 8) * 0.007),
        "hitter_whiff_rate_500_pa": str(0.17 + (index % 6) * 0.02),
        "hitter_k_rate_500_pa": str(0.15 + (index % 6) * 0.02),
        "hitter_whiff_rate_season": str(0.18 + (index % 6) * 0.02),
        "hitter_k_rate_season": str(0.16 + (index % 6) * 0.02),
        "hitter_split_ba_season_vs_lhp": ".280",
        "hitter_split_ba_season_vs_rhp": ".260",
        "hitter_split_pa_season_vs_lhp": "45",
        "hitter_split_pa_season_vs_rhp": "160",
        "hitter_split_ba_500_vs_lhp": ".270",
        "hitter_split_ba_500_vs_rhp": ".265",
        "pitcher_hand": "R",
        "pitcher_stuff_plus": str(88 + index % 20),
        "pitcher_hpi_200": "1.02",
        "pitcher_hpi_season": "1.08",
        "pitcher_hits_last_18_ip": "20",
        "inferred_pitch_type_xba": ".290",
        "h2h_pa": str(index % 8),
        "h2h_xba": ".310",
        "bullpen_hpi": "1.03",
        "bullpen_opp_ba": ".255",
        "park_hit_factor": "101",
        "forecast_temperature_f": "78",
        "hard_pass_reasons": "One | Two" if index % 5 == 0 else "",
        "road_game": "Y" if index % 2 else "N",
        "doubleheader": "N",
    }


def write_csv(path, rows):
    fields = sorted({field for item in rows for field in item})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_derived_features_use_provenance_and_shrink_rates():
    item = row(3, day=date(2026, 7, 1))
    values = derived_features(item)
    assert values["official_lineup"] == 1.0
    assert 0.25 < values["season_ba_shrunk"] < float(item["hitter_ba_season"])
    assert 0.26 <= values["matchup_hand_ba_shrunk"] <= 0.265
    assert values["stop_valve_count"] == 0
    assert "precip_probability" not in APPEARANCE_FEATURES + HIT_FEATURES
    assert "score" not in APPEARANCE_FEATURES + HIT_FEATURES


def test_no_appearance_is_negative_for_appearance_and_morning_outcome():
    item = row(1, day=date(2026, 7, 1), status="no_appearance")
    assert appearance_label(item) == 0
    assert decision_outcome(item) == 0


def test_historical_final_lineup_provenance_does_not_train_appearance_model():
    final_boxscore = row(1, day=date(2025, 7, 1))
    final_boxscore["lineup_source"] = "final_boxscore"
    assert appearance_label(final_boxscore) is None

    legacy_confirmed = row(2, day=date(2025, 7, 2))
    legacy_confirmed["lineup_source"] = ""
    assert appearance_label(legacy_confirmed) is None


def test_shadow_train_score_and_promotion_gate(tmp_path):
    start = date(2025, 4, 1)
    rows = []
    for index in range(480):
        day = start + timedelta(days=index // 8)
        status = "no_appearance" if index % 11 == 0 else "final"
        rows.append(row(index, day=day, status=status, hit=0 if index % 4 == 0 else 1))
    latest = start + timedelta(days=60)
    latest_rows = [row(1000 + index, day=latest, hit=index % 2) for index in range(8)]
    rows.extend(latest_rows)
    candidates = tmp_path / "candidates.csv"
    production = tmp_path / "production.csv"
    model_path = tmp_path / "shadow.json"
    report_path = tmp_path / "report.json"
    predictions_path = tmp_path / "predictions.csv"
    promotion_path = tmp_path / "promotion.json"
    write_csv(candidates, rows)
    write_csv(
        production,
        [
            {
                "date": item["date"],
                "player": item["player"],
                "player_id": item["player_id"],
                "team": item["team"],
                "opponent": item["opponent"],
                "game_pk": item["game_pk"],
                "learned_rank": str(index + 1),
                "learned_hit_probability": str(0.75 - index * 0.01),
            }
            for index, item in enumerate(latest_rows)
        ],
    )

    trained = train_shadow_model(
        candidates,
        production_predictions_csv=production,
        model_out=model_path,
        report_out=report_path,
        iterations=30,
    )
    assert trained["model"]["production_model_unchanged"] is True
    assert trained["model"]["calibrator"]["type"] in {"identity", "platt"}
    calibration = trained["report"]["calibration"]
    assert calibration["calibrator_date_max"] < calibration["date_min"]
    ranking = trained["report"]["untouched_ranking_validation"]
    assert ranking["resolved_top_one_days"] > 0
    assert 0 <= ranking["top_one_hit_rate"] <= 1
    deployment = trained["report"]["deployment_calibration"]
    assert deployment["training_date_max"] < deployment["calibrator_date_min"]
    assert deployment["holdout_dates"] == 14
    scored = score_shadow_candidates(
        candidates,
        production_predictions_csv=production,
        model_path=model_path,
        out_csv=predictions_path,
        date_filter=latest.isoformat(),
    )
    assert len(scored) == 8
    assert sorted(int(item["shadow_rank"]) for item in scored) == list(range(1, 9))
    assert len([item for item in scored if item["shadow_top5_rank"]]) == 5
    probabilities = [float(item["combined_probability_calibrated"]) for item in scored]
    assert all(0 < value < 1 for value in probabilities)

    report = promotion_report(
        candidates_csv=candidates,
        shadow_predictions_csv=predictions_path,
        out_json=promotion_path,
    )
    assert report["automatically_promoted"] is False
    assert report["minimum_resolved_days"] == 50
    assert json.loads(promotion_path.read_text())["status"] == "collecting"
