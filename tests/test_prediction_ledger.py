import csv
from pathlib import Path

import pytest

from statbirt.prediction_ledger import (
    LedgerConflictError,
    LedgerError,
    SNAPSHOT_FIELDS,
    audit_ledger,
    create_snapshot,
    main,
    sync_results,
)


TARGET_DATE = "2026-07-17"
SAVED_AT = "2026-07-17T12:00:00Z"
CUTOFF_AT = "2026-07-17T11:55:00Z"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _candidate_rows() -> list[dict[str, str]]:
    return [
        {
            "date": TARGET_DATE,
            "player": "First Hitter",
            "player_id": "101",
            "team": "CHC",
            "opponent": "STL",
            "game_pk": "9001",
            "game_start_time_utc": "2026-07-17T23:05:00Z",
            "confirmed_lineup": "Y",
            "result_hit": "",
            "result_hits": "",
            "result_ab": "",
            "result_pa": "",
            "result_status": "pending",
            "result_updated_at": "",
            "notes": "",
        },
        {
            "date": TARGET_DATE,
            "player": "Second Hitter",
            "player_id": "202",
            "team": "STL",
            "opponent": "CHC",
            "game_pk": "9001",
            "game_start_time_utc": "2026-07-17T23:05:00Z",
            "confirmed_lineup": "N",
            "result_hit": "",
            "result_hits": "",
            "result_ab": "",
            "result_pa": "",
            "result_status": "pending",
            "result_updated_at": "",
            "notes": "",
        },
    ]


def _production_rows() -> list[dict[str, str]]:
    return [
        {
            "date": TARGET_DATE,
            "player_id": "101",
            "game_pk": "9001",
            "model_version": "learned-v3-abc",
            "model_trained_at": "2026-07-17T10:00:00Z",
            "learned_hit_probability": "0.7000",
            "learned_rank": "1",
        },
        {
            "date": TARGET_DATE,
            "player_id": "202",
            "game_pk": "9001",
            "model_version": "learned-v3-abc",
            "model_trained_at": "2026-07-17T10:00:00Z",
            "learned_hit_probability": "0.60",
            "learned_rank": "2",
        },
    ]


def _shadow_rows() -> list[dict[str, str]]:
    return [
        {
            "date": TARGET_DATE,
            "player_id": "202",
            "game_pk": "9001",
            "model_version": "shadow-v1-def",
            "model_trained_at": "2026-07-17T11:00:00Z",
            "scored_at": "2026-07-17T11:30:00Z",
            "candidate_pool_source": "projected-board",
            "lineup_source": "recent_usage",
            "production_probability": "0.60",
            "appearance_probability": "0.80",
            "hit_given_appearance_probability": "0.57",
            "combined_probability_raw": "0.456",
            "combined_probability_calibrated": "0.45",
            "shadow_rank": "1",
            "shadow_top5_rank": "1",
            "stuff_preference_rank": "2",
        },
        {
            "date": TARGET_DATE,
            "player_id": "101",
            "game_pk": "9001",
            "model_version": "shadow-v1-def",
            "model_trained_at": "2026-07-17T11:00:00Z",
            "scored_at": "2026-07-17T11:30:00Z",
            "candidate_pool_source": "official-lineup-board",
            "lineup_source": "official",
            "production_probability": "0.7000",
            "appearance_probability": "0.90",
            "hit_given_appearance_probability": "0.66",
            "combined_probability_raw": "0.594",
            "combined_probability_calibrated": "0.59",
            "shadow_rank": "2",
            "shadow_top5_rank": "2",
            "stuff_preference_rank": "1",
        },
    ]


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    candidates = tmp_path / "candidates.csv"
    production = tmp_path / "production.csv"
    shadow = tmp_path / "shadow.csv"
    snapshots = tmp_path / "decision_ledger" / "snapshots.csv"
    results = tmp_path / "decision_ledger" / "results.csv"
    _write_csv(candidates, _candidate_rows())
    _write_csv(production, _production_rows())
    _write_csv(shadow, _shadow_rows())
    return candidates, production, shadow, snapshots, results


def _snapshot(tmp_path: Path, run_id: str = "morning-run") -> tuple[dict[str, object], tuple[Path, ...]]:
    candidates, production, shadow, snapshots, results = _inputs(tmp_path)
    report = create_snapshot(
        run_id=run_id,
        candidates_csv=candidates,
        production_predictions_csv=production,
        shadow_predictions_csv=shadow,
        snapshots_csv=snapshots,
        target_date=TARGET_DATE,
        cutoff_at=CUTOFF_AT,
        saved_at=SAVED_AT,
        candidate_pool_source="daily-pipeline",
    )
    return report, (candidates, production, shadow, snapshots, results)


def test_snapshot_captures_decision_fields_and_provenance(tmp_path):
    report, paths = _snapshot(tmp_path)
    snapshots = paths[3]

    assert report["appended_rows"] == 2
    assert report["idempotent"] is False
    rows = _read_csv(snapshots)
    assert list(rows[0]) == SNAPSHOT_FIELDS
    assert [row["candidate_key"] for row in rows] == [
        "2026-07-17|101|9001",
        "2026-07-17|202|9001",
    ]
    assert rows[0]["candidate_pool_source"] == "daily-pipeline"
    assert rows[0]["lineup_provenance"] == "official"
    assert rows[1]["lineup_provenance"] == "recent_usage"
    assert rows[0]["raw_probability"] == "0.7"
    assert rows[0]["calibrated_probability"] == "0.59"
    assert rows[0]["appearance_probability"] == "0.9"
    assert rows[0]["combined_probability"] == "0.59"
    assert rows[0]["hit_given_appearance_probability"] == "0.66"
    assert rows[0]["combined_probability_raw"] == "0.594"
    assert rows[0]["combined_probability_calibrated"] == "0.59"
    assert [row["shadow_rank"] for row in rows] == ["2", "1"]
    assert [row["shadow_top5_rank"] for row in rows] == ["2", "1"]
    assert [row["stuff_preference_rank"] for row in rows] == ["1", "2"]
    assert all(len(row["feature_hash"]) == 64 for row in rows)
    assert all(len(row["snapshot_hash"]) == 64 for row in rows)


def test_same_run_is_idempotent_new_run_appends_and_conflict_fails(tmp_path):
    _, paths = _snapshot(tmp_path)
    candidates, production, shadow, snapshots, _ = paths
    original = snapshots.read_bytes()

    rerun = create_snapshot(
        run_id="morning-run",
        candidates_csv=candidates,
        production_predictions_csv=production,
        shadow_predictions_csv=shadow,
        snapshots_csv=snapshots,
        target_date=TARGET_DATE,
        cutoff_at=CUTOFF_AT,
        saved_at=SAVED_AT,
        candidate_pool_source="daily-pipeline",
    )
    assert rerun["idempotent"] is True
    assert rerun["appended_rows"] == 0
    assert snapshots.read_bytes() == original

    appended = create_snapshot(
        run_id="lineup-run",
        candidates_csv=candidates,
        production_predictions_csv=production,
        shadow_predictions_csv=shadow,
        snapshots_csv=snapshots,
        target_date=TARGET_DATE,
        cutoff_at=CUTOFF_AT,
        saved_at=SAVED_AT,
        candidate_pool_source="daily-pipeline",
    )
    assert appended["appended_rows"] == 2
    assert len(_read_csv(snapshots)) == 4

    changed_predictions = _production_rows()
    changed_predictions[0]["learned_hit_probability"] = "0.71"
    _write_csv(production, changed_predictions)
    with pytest.raises(LedgerConflictError, match="different immutable snapshot"):
        create_snapshot(
            run_id="morning-run",
            candidates_csv=candidates,
            production_predictions_csv=production,
            shadow_predictions_csv=shadow,
            snapshots_csv=snapshots,
            target_date=TARGET_DATE,
            cutoff_at=CUTOFF_AT,
            saved_at=SAVED_AT,
            candidate_pool_source="daily-pipeline",
        )


def test_result_sync_is_separate_upsert_and_is_stable_when_unchanged(tmp_path):
    _, paths = _snapshot(tmp_path)
    candidates, _, _, snapshots, results = paths
    immutable_snapshot = snapshots.read_bytes()
    resolved = _candidate_rows()
    resolved[0].update(
        {
            "result_hit": "1",
            "result_hits": "1",
            "result_ab": "4",
            "result_pa": "5",
            "result_status": "Game Over",
            "result_updated_at": "2026-07-18T04:00:00Z",
        }
    )
    resolved[1].update(
        {
            "result_hit": "",
            "result_hits": "0",
            "result_ab": "0",
            "result_pa": "0",
            "result_status": "No Appearance",
            "result_updated_at": "2026-07-18T04:00:00Z",
        }
    )
    _write_csv(candidates, resolved)

    first = sync_results(
        candidates_csv=candidates,
        snapshots_csv=snapshots,
        results_csv=results,
        synced_at="2026-07-18T05:00:00Z",
    )
    assert first["inserted_rows"] == 2
    assert snapshots.read_bytes() == immutable_snapshot
    result_rows = _read_csv(results)
    assert [row["result_status"] for row in result_rows] == ["final", "no_appearance"]
    assert all(row["run_id"] == "morning-run" for row in result_rows)
    unchanged_bytes = results.read_bytes()

    second = sync_results(
        candidates_csv=candidates,
        snapshots_csv=snapshots,
        results_csv=results,
        synced_at="2026-07-18T06:00:00Z",
    )
    assert second["unchanged_rows"] == 2
    assert results.read_bytes() == unchanged_bytes

    resolved[0]["result_hits"] = "2"
    _write_csv(candidates, resolved)
    third = sync_results(
        candidates_csv=candidates,
        snapshots_csv=snapshots,
        results_csv=results,
        synced_at="2026-07-18T07:00:00Z",
    )
    assert third["updated_rows"] == 1
    assert len(_read_csv(results)) == 2
    assert _read_csv(results)[0]["result_hits"] == "2"
    assert snapshots.read_bytes() == immutable_snapshot
    assert audit_ledger(snapshots_csv=snapshots, results_csv=results)["ok"] is True


def test_audit_detects_snapshot_content_tampering(tmp_path):
    _, paths = _snapshot(tmp_path)
    snapshots = paths[3]
    rows = _read_csv(snapshots)
    rows[0]["raw_probability"] = "0.1"
    _write_csv(snapshots, rows)

    report = audit_ledger(snapshots_csv=snapshots, results_csv=paths[4])
    assert report["ok"] is False
    assert any("snapshot_hash" in error for error in report["errors"])


def test_snapshot_rejects_an_incomplete_shadow_decision(tmp_path):
    candidates, production, shadow, snapshots, _ = _inputs(tmp_path)
    incomplete = _shadow_rows()
    incomplete[0]["combined_probability_calibrated"] = ""
    incomplete[0]["combined_probability_raw"] = ""
    _write_csv(shadow, incomplete)

    with pytest.raises(LedgerError, match="calibrated_probability must not be blank"):
        create_snapshot(
            run_id="incomplete-run",
            candidates_csv=candidates,
            production_predictions_csv=production,
            shadow_predictions_csv=shadow,
            snapshots_csv=snapshots,
            target_date=TARGET_DATE,
            cutoff_at=CUTOFF_AT,
            saved_at=SAVED_AT,
        )


def test_snapshot_rejects_shadow_scored_after_first_pitch(tmp_path):
    candidates, production, shadow, snapshots, _ = _inputs(tmp_path)
    late_shadow = _shadow_rows()
    for item in late_shadow:
        item["scored_at"] = "2026-07-17T23:10:00Z"
    _write_csv(shadow, late_shadow)

    with pytest.raises(LedgerError, match="shadow_scored_at is not pregame"):
        create_snapshot(
            run_id="late-shadow-run",
            candidates_csv=candidates,
            production_predictions_csv=production,
            shadow_predictions_csv=shadow,
            snapshots_csv=snapshots,
            target_date=TARGET_DATE,
            cutoff_at=CUTOFF_AT,
            saved_at=SAVED_AT,
        )


def test_cli_snapshot_sync_results_and_audit(tmp_path, capsys):
    candidates, production, shadow, snapshots, results = _inputs(tmp_path)
    assert (
        main(
            [
                "snapshot",
                "--run-id",
                "cli-run",
                "--target-date",
                TARGET_DATE,
                "--cutoff-at",
                CUTOFF_AT,
                "--saved-at",
                SAVED_AT,
                "--candidates",
                str(candidates),
                "--production-predictions",
                str(production),
                "--shadow-predictions",
                str(shadow),
                "--snapshots",
                str(snapshots),
                "--candidate-pool-source",
                "cli-test",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "sync-results",
                "--candidates",
                str(candidates),
                "--snapshots",
                str(snapshots),
                "--results",
                str(results),
                "--synced-at",
                "2026-07-18T05:00:00Z",
            ]
        )
        == 0
    )
    assert main(["audit", "--snapshots", str(snapshots), "--results", str(results)]) == 0
    assert '"ok": true' in capsys.readouterr().out
