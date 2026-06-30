import csv
import json

import pytest

from statbirt.stage_learned_thesis import build_thesis_context, stage_thesis_context


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_thesis_context_stages_learned_top_two(tmp_path):
    candidates_csv = tmp_path / "candidates.csv"
    predictions_csv = tmp_path / "predictions.csv"
    congregation_csv = tmp_path / "congregation.csv"

    write_rows(
        candidates_csv,
        [
            "date",
            "player",
            "player_id",
            "team",
            "opponent",
            "game_pk",
            "venue_name",
            "pickable",
            "score",
            "hard_pass_reasons",
            "concerns",
            "expected_pa",
            "lineup_slot",
            "hitter_ba_season",
            "hitter_last_5_games_played",
            "hitter_last_5_games_hits",
            "hitter_last_5_games_ab",
            "hitter_last_5_games_ba",
            "hitter_k_rate_500_pa",
            "h2h_pa",
            "h2h_hits",
            "h2h_hit_rate",
            "probable_pitcher",
            "probable_pitcher_id",
            "pitcher_hand",
            "batter_stand",
            "pitcher_stuff_plus",
            "pitcher_last_start_ip",
            "pitcher_last_start_hits",
            "pitcher_last_start_strikeouts",
            "pitcher_last_start_walks",
            "bullpen_hpi",
            "bullpen_opp_ba",
        ],
        [
            {
                "date": "2026-06-01",
                "player": "Alpha Hitter",
                "player_id": "111",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "9001",
                "venue_name": "Test Park",
                "pickable": "N",
                "score": "70.0",
                "hard_pass_reasons": "Season K >22%",
                "concerns": "Watch bullpen",
                "expected_pa": "4.6",
                "lineup_slot": "2",
                "hitter_ba_season": "0.301",
                "hitter_last_5_games_played": "5",
                "hitter_last_5_games_hits": "7",
                "hitter_last_5_games_ab": "15",
                "hitter_last_5_games_ba": "0.467",
                "hitter_k_rate_500_pa": "0.210",
                "h2h_pa": "8",
                "h2h_hits": "3",
                "h2h_hit_rate": "0.375",
                "probable_pitcher": "Starter One",
                "probable_pitcher_id": "222",
                "pitcher_hand": "R",
                "batter_stand": "L",
                "pitcher_stuff_plus": "91",
                "pitcher_last_start_ip": "6.0",
                "pitcher_last_start_hits": "3",
                "pitcher_last_start_strikeouts": "8",
                "pitcher_last_start_walks": "2",
                "bullpen_hpi": "1.05",
                "bullpen_opp_ba": "0.265",
            },
            {
                "date": "2026-06-01",
                "player": "Beta Hitter",
                "player_id": "333",
                "team": "CCC",
                "opponent": "DDD",
                "game_pk": "9002",
                "venue_name": "Other Park",
                "pickable": "Y",
                "score": "68.0",
                "hard_pass_reasons": "",
                "concerns": "",
                "expected_pa": "4.4",
                "lineup_slot": "3",
                "hitter_ba_season": "0.288",
                "hitter_last_5_games_played": "5",
                "hitter_last_5_games_hits": "5",
                "hitter_last_5_games_ab": "16",
                "hitter_last_5_games_ba": "0.313",
                "hitter_k_rate_500_pa": "0.180",
                "h2h_pa": "4",
                "h2h_hits": "1",
                "h2h_hit_rate": "0.250",
                "probable_pitcher": "Starter Two",
                "probable_pitcher_id": "444",
                "pitcher_hand": "L",
                "batter_stand": "R",
                "pitcher_stuff_plus": "94",
                "pitcher_last_start_ip": "5.2",
                "pitcher_last_start_hits": "5",
                "pitcher_last_start_strikeouts": "5",
                "pitcher_last_start_walks": "1",
                "bullpen_hpi": "0.98",
                "bullpen_opp_ba": "0.241",
            },
        ],
    )
    write_rows(
        predictions_csv,
        [
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
            "result_status",
        ],
        [
            {
                "date": "2026-06-01",
                "player": "Beta Hitter",
                "player_id": "333",
                "team": "CCC",
                "opponent": "DDD",
                "game_pk": "9002",
                "bob_score": "68.0",
                "pickable": "Y",
                "learned_hit_probability": "0.7600",
                "learned_rank": "2",
                "model_version": "learned-test",
                "model_trained_at": "2026-06-01T12:00:00Z",
            },
            {
                "date": "2026-06-01",
                "player": "Alpha Hitter",
                "player_id": "111",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "9001",
                "bob_score": "70.0",
                "pickable": "N",
                "learned_hit_probability": "0.8100",
                "learned_rank": "1",
                "model_version": "learned-test",
                "model_trained_at": "2026-06-01T12:00:00Z",
            },
        ],
    )
    write_rows(
        congregation_csv,
        ["player", "player_id", "status", "aliases"],
        [{"player": "Alpha Hitter", "player_id": "111", "status": "Publisher", "aliases": ""}],
    )

    payload = build_thesis_context(
        predictions_csv=predictions_csv,
        candidates_csv=candidates_csv,
        congregation_csv=congregation_csv,
        target_date="latest",
        top=2,
        filter_injured=False,
        thesis_dir=tmp_path / "theses",
    )

    assert payload["date"] == "2026-06-01"
    assert payload["target"] == "learned_rank_top_2"
    assert payload["expected_thesis_path"].endswith("2026-06-01.json")
    assert [player["player"] for player in payload["players"]] == ["Alpha Hitter", "Beta Hitter"]
    first = payload["players"][0]
    assert first["dashboard_summary"]["congregation_status"] == "Publisher"
    assert first["dashboard_summary"]["hot_streak"] is True
    assert first["field_context"]["h2h"]["h2h_pa"] == "8"
    assert first["field_context"]["probable_starter_last_start"]["pitcher_last_start_strikeouts"] == "8"
    assert any(source["label"] == "Baseball Savant: Alpha Hitter" for source in first["source_targets"])
    assert payload["selection_brief_from_local_metrics"]["recommended_single"]


def test_stage_thesis_context_writes_json(tmp_path):
    candidates_csv = tmp_path / "candidates.csv"
    predictions_csv = tmp_path / "predictions.csv"
    congregation_csv = tmp_path / "congregation.csv"

    write_rows(
        candidates_csv,
        ["date", "player", "player_id", "team", "opponent", "game_pk", "pickable", "score"],
        [{"date": "2026-06-01", "player": "Alpha Hitter", "player_id": "111", "team": "AAA", "opponent": "BBB", "game_pk": "9001", "pickable": "Y", "score": "70"}],
    )
    write_rows(
        predictions_csv,
        ["date", "player", "player_id", "team", "opponent", "game_pk", "bob_score", "pickable", "learned_hit_probability", "learned_rank", "model_version", "model_trained_at"],
        [{"date": "2026-06-01", "player": "Alpha Hitter", "player_id": "111", "team": "AAA", "opponent": "BBB", "game_pk": "9001", "bob_score": "70", "pickable": "Y", "learned_hit_probability": "0.8100", "learned_rank": "1", "model_version": "learned-test", "model_trained_at": "2026-06-01T12:00:00Z"}],
    )
    write_rows(congregation_csv, ["player", "player_id", "status", "aliases"], [])

    payload, output_path = stage_thesis_context(
        predictions_csv=predictions_csv,
        candidates_csv=candidates_csv,
        congregation_csv=congregation_csv,
        target_date="2026-06-01",
        top=1,
        filter_injured=False,
        out_dir=tmp_path / "context",
        thesis_dir=tmp_path / "theses",
    )

    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["date"] == payload["date"] == "2026-06-01"


def test_build_thesis_context_rejects_empty_top(tmp_path):
    with pytest.raises(ValueError, match="top"):
        build_thesis_context(
            predictions_csv=tmp_path / "missing_predictions.csv",
            candidates_csv=tmp_path / "missing_candidates.csv",
            target_date="latest",
            top=0,
            filter_injured=False,
        )
