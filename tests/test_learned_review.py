from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from statbirt.learned_review import calibration_summary, prospective_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prospective_mask_requires_training_before_first_pitch():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "model_trained_at": "2026-07-01T12:00:00Z",
                "game_start_time_utc": "2026-07-01T23:00:00Z",
            },
            {
                "date": "2026-07-01",
                "model_trained_at": "2026-07-02T12:00:00Z",
                "game_start_time_utc": "2026-07-01T23:00:00Z",
            },
            {
                "date": "2026-07-01",
                "model_trained_at": "2026-07-01T12:00:00Z",
                "game_start_time_utc": "",
            },
        ]
    )

    assert prospective_mask(frame).tolist() == [True, False, True]


def test_calibration_summary_uses_only_resolved_rows():
    frame = pd.DataFrame(
        [
            {"learned_hit_probability": 0.72, "label": 1},
            {"learned_hit_probability": 0.74, "label": 0},
            {"learned_hit_probability": 0.76, "label": 1},
            {"learned_hit_probability": 0.77, "label": None},
        ]
    )

    rows = {row["probability_band"]: row for row in calibration_summary(frame)}

    assert rows["70-75%"]["decisions"] == 2
    assert rows["70-75%"]["hits"] == 1
    assert rows["75-80%"]["decisions"] == 1
    assert rows["75-80%"]["hits"] == 1


def test_both_dashboards_link_the_dated_learned_review():
    expected_href = "data/learned_model_review.html"
    for name in ("index.html", "learned.html"):
        soup = BeautifulSoup((PROJECT_ROOT / "web" / name).read_text(encoding="utf-8"), "html.parser")
        links = [link for link in soup.select("nav.dashboard-switcher a") if link.get("href") == expected_href]
        assert len(links) == 1
        assert "7-16-26 Learned Model Review" in links[0].get_text(" ", strip=True)

    assert (PROJECT_ROOT / "web" / expected_href).exists()
