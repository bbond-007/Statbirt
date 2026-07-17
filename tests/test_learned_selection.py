from statbirt.learned_selection import backtest_pick_payloads, build_selection_brief, prediction_is_pregame


def pick(name, rank, probability, *, hit=False, safety=75, h2h_pa=4, h2h_hits=2):
    return {
        "player": name,
        "rank": rank,
        "learned_rank": rank,
        "learned_hit_probability": probability,
        "safety_score": safety,
        "bob_score": 60,
        "expected_pa": "4.5",
        "lineup_slot": "2",
        "hitter_ba_season": "0.300",
        "h2h_pa": h2h_pa,
        "h2h_hits": h2h_hits,
        "h2h_record": f"{h2h_hits}-{h2h_pa}",
        "selection_signals": ["Expected PA 4.5"],
        "selection_risks": [],
        "hard_pass_reasons": ["Example stop valve"],
        "result_status": "final",
        "result_hit": hit,
        "result_hits": 1 if hit else 0,
    }


def test_build_selection_brief_recommends_single_and_pair():
    picks = [
        pick("A", 1, 0.70, safety=70),
        pick("B", 2, 0.69, safety=90),
        pick("C", 3, 0.68, safety=65, h2h_pa=0, h2h_hits=0),
    ]

    brief = build_selection_brief(picks)

    assert brief["recommended_single"]["player"] in {"A", "B"}
    assert len(brief["recommended_pair"]) == 2
    assert brief["items"][0]["pros"]
    assert brief["items"][0]["cons"]
    assert brief["items"][0]["thesis"]
    assert brief["items"][0]["confidence_label"] in {"Primary", "Strong", "Playable", "Watch"}
    assert brief["items"][0]["hard_pass_reasons"] == ["Example stop valve"]
    assert brief["items"][0]["result_status"] == "final"


def test_backtest_pick_payloads_tracks_top1_and_top2_any_hit():
    payload = backtest_pick_payloads(
        {
            "2026-05-01": [
                pick("A", 1, 0.70, hit=False),
                pick("B", 2, 0.69, hit=True, safety=90),
            ],
            "2026-05-02": [
                pick("C", 1, 0.72, hit=True),
                pick("D", 2, 0.68, hit=False),
            ],
        }
    )

    assert payload["days"] == 2
    assert payload["top2_any_hit_rate"] == 1.0
    assert payload["top2_any_longest_streak"] == 2
    assert payload["daily"][0]["pair_any_hit"] is True


def test_backtest_pick_payloads_excludes_no_appearance_rows():
    no_appearance = pick("A", 1, 0.70, hit=False)
    no_appearance["result_status"] = "no_appearance"
    payload = backtest_pick_payloads(
        {
            "2026-05-01": [
                no_appearance,
                pick("B", 2, 0.69, hit=True),
            ]
        }
    )

    assert payload["days"] == 0
    assert payload["top1_hit_rate"] is None
    assert payload["top2_any_hit_rate"] is None
    assert payload["daily"] == []


def test_prediction_is_pregame_uses_first_pitch_when_available():
    prediction = {"date": "2026-07-01", "model_trained_at": "2026-07-01T12:00:00Z"}

    assert prediction_is_pregame(prediction, {"game_start_time_utc": "2026-07-01T23:00:00Z"})
    assert not prediction_is_pregame(prediction, {"game_start_time_utc": "2026-07-01T11:00:00Z"})
    assert not prediction_is_pregame(
        {"date": "2025-07-01", "model_trained_at": "2026-05-15T12:00:00Z"},
        {"date": "2025-07-01", "game_start_time_utc": ""},
    )
