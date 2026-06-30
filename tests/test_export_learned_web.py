import csv
import json

from statbirt.export_learned_web import build_pick_payload, export_learned_web_payload
from statbirt.export_web import export_web_payload
from statbirt.injuries import is_injured_status


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_export_learned_web_joins_predictions_to_candidates(tmp_path):
    candidates_csv = tmp_path / "candidates.csv"
    predictions_csv = tmp_path / "predictions.csv"
    congregation_csv = tmp_path / "congregation.csv"
    out_json = tmp_path / "learned_shortlist.json"
    index_json = tmp_path / "learned_dashboard_index.json"
    archive_dir = tmp_path / "learned_dashboards"
    top2_thesis_dir = tmp_path / "learned_top2_theses"

    write_rows(
        candidates_csv,
        [
            "date",
            "player",
            "player_id",
            "team",
            "opponent",
            "game_pk",
            "pickable",
            "score",
            "hard_pass_reasons",
            "expected_pa",
            "hitter_pa_per_game_season",
            "hitter_ba_season",
            "lineup_slot",
            "hitter_k_rate_500_pa",
            "hitter_hipa_500_pa",
            "pitcher_stuff_plus",
            "result_status",
        ],
        [
            {
                "date": "2026-05-15",
                "player": "Test Hitter",
                "player_id": "123",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "9001",
                "pickable": "Y",
                "score": "61.5",
                "hard_pass_reasons": "",
                "expected_pa": "4.6",
                "hitter_pa_per_game_season": "4.5",
                "hitter_ba_season": "0.312",
                "lineup_slot": "2",
                "hitter_k_rate_500_pa": "0.18",
                "hitter_hipa_500_pa": "0.291",
                "pitcher_stuff_plus": "91",
                "result_status": "final",
            }
        ],
    )
    write_rows(
        congregation_csv,
        ["player", "player_id", "status", "aliases"],
        [
            {
                "player": "Test Hitter",
                "player_id": "123",
                "status": "Publisher",
                "aliases": "",
            }
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
                "date": "2026-05-15",
                "player": "Test Hitter",
                "player_id": "123",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "9001",
                "bob_score": "61.5",
                "pickable": "Y",
                "learned_hit_probability": "0.8123",
                "learned_rank": "1",
                "model_version": "learned-test",
                "model_trained_at": "2026-05-15T12:00:00Z",
                "result_hit": "1",
                "result_hits": "2",
                "result_ab": "4",
                "result_pa": "4",
                "result_status": "final",
            }
        ],
    )
    top2_thesis_dir.mkdir()
    (top2_thesis_dir / "2026-05-15.json").write_text(
        json.dumps(
            {
                "date": "2026-05-15",
                "target": "learned_rank_top_2",
                "committee_pick": "Test Hitter",
                "committee_summary": "The committee prefers Test Hitter.",
                "players": [
                    {
                        "rank": 1,
                        "player": "Test Hitter",
                        "pro_thesis": "Strong opportunity and form.",
                        "con_thesis": "One-game outcomes are fragile.",
                        "committee_thesis": "The committee still prefers this profile.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = export_learned_web_payload(
        predictions_csv=predictions_csv,
        candidates_csv=candidates_csv,
        congregation_csv=congregation_csv,
        out_json=out_json,
        index_json=index_json,
        archive_dir=archive_dir,
        target_date="latest",
        limit=1,
        top2_thesis_dir=top2_thesis_dir,
    )

    assert payload["requested_date"] == "2026-05-15"
    assert payload["date"] == "2026-05-15"
    assert payload["model_version"] == "learned-test"
    assert payload["top_5_hit_count"] == 1
    assert payload["picks"][0]["matched_candidate"] is True
    assert payload["picks"][0]["learned_hit_probability"] == 0.8123
    assert payload["picks"][0]["hitter_ba_season"] == "0.312"
    assert payload["picks"][0]["congregation_status"] == "Publisher"
    assert payload["picks"][0]["congregation_member"] is True
    assert payload["picks"][0]["result_hit"] is True
    assert payload["picks"][0]["safety_score"] > 80
    assert payload["daily_selection_brief"]["recommended_single"]["player"] == "Test Hitter"
    assert payload["daily_selection_brief"]["items"][0]["pros"]
    assert payload["daily_selection_brief"]["items"][0]["cons"]
    assert payload["learned_top2_thesis"]["committee_pick"] == "Test Hitter"
    assert payload["learned_top2_thesis"]["players"][0]["pro_thesis"]
    assert out_json.exists()
    assert index_json.exists()
    assert (archive_dir / "2026-05-15.json").exists()


def test_export_learned_web_prefers_candidate_results_when_predictions_are_blank(tmp_path):
    candidates_csv = tmp_path / "candidates.csv"
    predictions_csv = tmp_path / "predictions.csv"

    write_rows(
        candidates_csv,
        ["date", "player", "player_id", "team", "opponent", "game_pk", "pickable", "score", "result_hit", "result_hits", "result_ab", "result_pa", "result_status"],
        [
            {
                "date": "2026-05-15",
                "player": "Fresh Result",
                "player_id": "123",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "9001",
                "pickable": "Y",
                "score": "70.0",
                "result_hit": "1",
                "result_hits": "2",
                "result_ab": "4",
                "result_pa": "4",
                "result_status": "final",
            }
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
                "date": "2026-05-15",
                "player": "Fresh Result",
                "player_id": "123",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "9001",
                "bob_score": "70.0",
                "pickable": "Y",
                "learned_hit_probability": "0.8000",
                "learned_rank": "1",
                "model_version": "learned-test",
                "model_trained_at": "2026-05-15T12:00:00Z",
                "result_hit": "",
                "result_hits": "",
                "result_ab": "",
                "result_pa": "",
                "result_status": "",
            }
        ],
    )

    payload = export_learned_web_payload(
        predictions_csv=predictions_csv,
        candidates_csv=candidates_csv,
        out_json=tmp_path / "learned_shortlist.json",
        index_json=tmp_path / "learned_dashboard_index.json",
        archive_dir=tmp_path / "learned_dashboards",
        target_date="2026-05-15",
        limit=1,
        archive=False,
    )

    pick = payload["picks"][0]
    assert pick["result_status"] == "final"
    assert pick["result_hit"] is True
    assert pick["result_hits"] == 2
    assert payload["top_5_decided_count"] == 1
    assert payload["top_5_hit_count"] == 1


def test_learned_pick_payload_uses_final_game_state_when_results_are_blank():
    prediction = {
        "date": "2026-05-15",
        "player": "Live Lookup",
        "player_id": "123",
        "team": "AAA",
        "opponent": "BBB",
        "game_pk": "9001",
        "bob_score": "70.0",
        "pickable": "Y",
        "learned_hit_probability": "0.8000",
        "learned_rank": "1",
        "model_version": "learned-test",
        "model_trained_at": "2026-05-15T12:00:00Z",
        "result_hit": "",
        "result_hits": "",
        "result_ab": "",
        "result_pa": "",
        "result_status": "",
    }

    payload = build_pick_payload(
        prediction,
        {},
        1,
        game_states={
            (9001, 123): {
                "state": "hit",
                "status": "Final",
                "hits": 2,
            }
        },
    )

    assert payload["result_status"] == "final"
    assert payload["result_hit"] is True
    assert payload["result_hits"] == 2
    assert payload["game_state"] == "hit"


def test_injury_status_detection_uses_mlb_roster_statuses():
    assert is_injured_status({"code": "D10", "description": "Injured 10-Day"})
    assert is_injured_status({"code": "A", "description": "Injured 15-Day"})
    assert not is_injured_status({"code": "A", "description": "Active"})


def test_export_web_filters_currently_injured_players(tmp_path):
    candidates_csv = tmp_path / "candidates.csv"
    out_json = tmp_path / "top_picks.json"
    index_json = tmp_path / "dashboard_index.json"

    write_rows(
        candidates_csv,
        ["date", "player", "player_id", "team", "opponent", "game_pk", "pickable", "score", "h2h_pa", "h2h_hits"],
        [
            {
                "date": "2026-05-15",
                "player": "Healthy Hitter",
                "player_id": "123",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "",
                "pickable": "Y",
                "score": "70.0",
                "h2h_pa": "4",
                "h2h_hits": "1",
            },
            {
                "date": "2026-05-15",
                "player": "Injured Hitter",
                "player_id": "999",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "",
                "pickable": "Y",
                "score": "90.0",
                "h2h_pa": "8",
                "h2h_hits": "3",
            },
        ],
    )

    payload = export_web_payload(
        candidates_csv=candidates_csv,
        out_json=out_json,
        index_json=index_json,
        archive_dir=tmp_path / "dashboards",
        target_date="2026-05-15",
        limit=10,
        archive=False,
        injured_player_ids={999},
    )

    assert payload["injury_filtered_count"] == 1
    assert payload["total_candidates"] == 1
    assert [pick["player"] for pick in payload["picks"]] == ["Healthy Hitter"]


def test_export_learned_web_filters_currently_injured_players(tmp_path):
    candidates_csv = tmp_path / "candidates.csv"
    predictions_csv = tmp_path / "predictions.csv"

    write_rows(
        candidates_csv,
        ["date", "player", "player_id", "team", "opponent", "game_pk", "team_id", "pickable", "score"],
        [
            {
                "date": "2026-05-15",
                "player": "Healthy Hitter",
                "player_id": "123",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "",
                "team_id": "1",
                "pickable": "Y",
                "score": "70.0",
            },
            {
                "date": "2026-05-15",
                "player": "Injured Hitter",
                "player_id": "999",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "",
                "team_id": "1",
                "pickable": "Y",
                "score": "90.0",
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
                "date": "2026-05-15",
                "player": "Injured Hitter",
                "player_id": "999",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "",
                "bob_score": "90.0",
                "pickable": "Y",
                "learned_hit_probability": "0.9000",
                "learned_rank": "1",
                "model_version": "learned-test",
                "model_trained_at": "2026-05-15T12:00:00Z",
                "result_hit": "",
                "result_hits": "",
                "result_ab": "",
                "result_pa": "",
                "result_status": "pending",
            },
            {
                "date": "2026-05-15",
                "player": "Healthy Hitter",
                "player_id": "123",
                "team": "AAA",
                "opponent": "BBB",
                "game_pk": "",
                "bob_score": "70.0",
                "pickable": "Y",
                "learned_hit_probability": "0.8000",
                "learned_rank": "2",
                "model_version": "learned-test",
                "model_trained_at": "2026-05-15T12:00:00Z",
                "result_hit": "",
                "result_hits": "",
                "result_ab": "",
                "result_pa": "",
                "result_status": "pending",
            },
        ],
    )

    payload = export_learned_web_payload(
        predictions_csv=predictions_csv,
        candidates_csv=candidates_csv,
        out_json=tmp_path / "learned_shortlist.json",
        index_json=tmp_path / "learned_dashboard_index.json",
        archive_dir=tmp_path / "learned_dashboards",
        target_date="2026-05-15",
        limit=5,
        archive=False,
        injured_player_ids={999},
    )

    assert payload["injury_filtered_count"] == 1
    assert payload["total_predictions"] == 1
    assert [pick["player"] for pick in payload["picks"]] == ["Healthy Hitter"]
