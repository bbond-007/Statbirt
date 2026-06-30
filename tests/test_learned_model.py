import csv
from pathlib import Path

from statbirt.learned_model import score_candidates, train_model


def _write_candidates(path: Path) -> None:
    fieldnames = [
        "date",
        "player",
        "player_id",
        "team",
        "opponent",
        "game_pk",
        "score",
        "pickable",
        "expected_pa",
        "hitter_hipa_500_pa",
        "pitcher_hpi_season",
        "bullpen_hpi",
        "road_game",
        "hard_pass_reasons",
        "concerns",
        "result_hit",
        "result_hits",
        "result_ab",
        "result_pa",
        "result_status",
    ]
    rows = []
    for idx in range(8):
        hit = "1" if idx % 2 == 0 else "0"
        rows.append(
            {
                "date": f"2026-04-{26 + idx // 2:02d}",
                "player": f"Hitter {idx}",
                "player_id": str(1000 + idx),
                "team": "BOS" if idx % 2 == 0 else "NYY",
                "opponent": "TOR",
                "game_pk": str(9000 + idx),
                "score": str(60 + idx),
                "pickable": "Y" if idx % 2 == 0 else "N",
                "expected_pa": "4.7" if idx % 2 == 0 else "3.8",
                "hitter_hipa_500_pa": "0.290" if idx % 2 == 0 else "0.210",
                "pitcher_hpi_season": "1.10" if idx % 2 == 0 else "0.82",
                "bullpen_hpi": "1.05",
                "road_game": "Y" if idx % 2 == 0 else "N",
                "hard_pass_reasons": "" if idx % 2 == 0 else "Starter Stuff+ over 95",
                "concerns": "",
                "result_hit": hit,
                "result_hits": hit,
                "result_ab": "4",
                "result_pa": "4",
                "result_status": "Game Over",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_train_and_score_learned_model(tmp_path):
    candidates = tmp_path / "candidates.csv"
    model = tmp_path / "model.json"
    report = tmp_path / "report.json"
    predictions = tmp_path / "predictions.csv"
    _write_candidates(candidates)

    result = train_model(
        candidates,
        model_out=model,
        report_out=report,
        min_rows=4,
        iterations=50,
        learning_rate=0.05,
    )
    assert model.exists()
    assert report.exists()
    assert result["report"]["labeled_training_rows"] == 8

    records = score_candidates(candidates, model_path=model, out_csv=predictions, date_filter="latest")
    assert predictions.exists()
    assert len(records) == 2
    assert records[0]["learned_rank"] == "1"
    assert 0.0 <= float(records[0]["learned_hit_probability"]) <= 1.0
    assert records[0]["result_status"] == "final"
    with predictions.open(newline="", encoding="utf-8") as f:
        prediction_row = next(csv.DictReader(f))
    assert prediction_row["result_status"] == "final"
