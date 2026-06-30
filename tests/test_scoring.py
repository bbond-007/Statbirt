from datetime import date, datetime, timezone
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pandas as pd

from statbirt import export_web
from statbirt import results as results_module
from statbirt import update_bullpen as bullpen_update
from statbirt import update_weather as weather_update
from statbirt import weather
from statbirt.export_web import candidate_payload, is_roofed_ballpark
from statbirt.mlb_api import (
    BatterUsageEntry,
    HitterPlayEntry,
    compute_hitter_windows,
    get_games_for_date,
    parse_pitcher_game_entries,
    pitcher_last_start_stats,
)
from statbirt.models import CandidateFeatures
from statbirt.pipeline import recent_batting_summary, scored_candidate_to_row
from statbirt.results import (
    RESULT_STATUS_FINAL,
    RESULT_STATUS_NO_APPEARANCE,
    RESULT_STATUS_PENDING,
    RESULT_STATUS_POSTPONED,
    has_result_data,
    normalize_row_result_status,
    result_status_for_game_status,
    update_results_csv,
    upsert_candidate_rows,
)
from statbirt.savant import StatcastFeatureStore
from statbirt.scoring import evaluate_stop_valves, expected_pa_score, score_candidate, score_features


def base_features(**overrides):
    values = {
        "target_date": date(2026, 4, 26),
        "player_id": 1,
        "player_name": "Test Hitter",
        "team_id": 111,
        "team": "BOS",
        "opponent_id": 147,
        "opponent": "NYY",
        "game_pk": 1,
        "is_home": False,
        "lineup_slot": 2.0,
        "starts_last_5": 5,
        "pitcher_id": 2,
        "pitcher_name": "Test Pitcher",
        "pitcher_hand": "R",
        "batter_stand": "L",
        "same_division": True,
        "hitter_hipa_2500_pa": 0.230,
        "hitter_pa_per_game_season": 4.45,
        "hitter_ba_season": 0.305,
        "hitter_ba_2500_ab": 0.295,
        "hitter_hipa_500_pa": 0.240,
        "hitter_hipa_75_ab": 0.260,
        "hitter_ba_75_ab": 0.320,
        "hitter_ba_25_ab": 0.240,
        "hitter_ba_500_ab": 0.285,
        "hitter_bb_rate_season": 0.080,
        "hitter_bb_rate_500_pa": 0.090,
        "hitter_whiff_rate_season": 0.180,
        "hitter_whiff_rate_500_pa": 0.190,
        "hitter_k_rate_season": 0.170,
        "hitter_k_rate_500_pa": 0.180,
        "hitter_split_ba_500_vs_lhp": 0.270,
        "hitter_split_ba_500_vs_rhp": 0.280,
        "hitter_split_ba_1500_vs_lhp": 0.268,
        "hitter_split_ba_1500_vs_rhp": 0.282,
        "hitter_matchup_hand_ba_500": 0.280,
        "hitter_matchup_hand_ba_1500": 0.275,
        "pitcher_hpi_350": 1.050,
        "pitcher_hpi_200": 0.940,
        "pitcher_hpi_season": 1.000,
        "pitcher_hits_last_18_ip": 18,
        "pitcher_last_start_date": date(2026, 4, 20),
        "pitcher_last_start_ip": 5.2,
        "pitcher_last_start_hits": 5,
        "pitcher_last_start_strikeouts": 7,
        "pitcher_last_start_walks": 1,
        "pitcher_stuff_plus": 92.0,
        "h2h_pa": 4,
        "h2h_hit_rate": 0.300,
        "h2h_whiff_rate": 0.150,
        "h2h_k_rate": 0.100,
        "h2h_exit_velocity": 90.0,
        "h2h_xba": 0.310,
        "pitcher_lr_opp_ba": 0.275,
        "pitcher_lr_opp_ba_50": 0.260,
        "pitcher_lr_opp_ba_200": 0.255,
        "inferred_pitch_type_ba": 0.290,
        "inferred_pitch_type_xba": 0.300,
        "inferred_pitch_type_coverage": 0.65,
        "bullpen_hpi": 1.020,
        "sprint_speed": 28.0,
        "park_hit_factor": 103.0,
        "expected_pa": 4.7,
        "precipitation_probability": 10.0,
    }
    values.update(overrides)
    return CandidateFeatures(**values)


def test_score_components_sum_to_weighted_score():
    score, components = score_features(base_features())
    assert round(sum(component.points for component in components), 2) == score
    assert 0 <= score <= 100
    component_names = {component.name for component in components}
    assert "hitter.pa_per_game_season" in component_names
    assert "hitter.hipa_75_ab" in component_names


def test_lower_lineup_slot_is_better_when_expected_pa_missing():
    assert expected_pa_score(1.0, None) == 100.0
    assert expected_pa_score(9.0, None) == 0.0


def test_recent_batting_summary_uses_last_five_games_played():
    entries = [
        BatterUsageEntry(date(2026, 4, 19), True, 1, hits=4, at_bats=4, plate_appearances=4),
        BatterUsageEntry(date(2026, 4, 20), True, 1, hits=1, at_bats=3, plate_appearances=4),
        BatterUsageEntry(date(2026, 4, 21), True, 1, hits=0, at_bats=2, plate_appearances=4),
        BatterUsageEntry(date(2026, 4, 22), True, 1, hits=2, at_bats=4, plate_appearances=4),
        BatterUsageEntry(date(2026, 4, 23), True, 1, hits=0, at_bats=0, plate_appearances=1),
        BatterUsageEntry(date(2026, 4, 24), True, 1, hits=1, at_bats=2, plate_appearances=4),
        BatterUsageEntry(date(2026, 4, 25), True, 1, hits=2, at_bats=2, plate_appearances=4),
        BatterUsageEntry(date(2026, 4, 26), True, 1, hits=3, at_bats=4, plate_appearances=4),
    ]

    summary = recent_batting_summary(entries, target_date=date(2026, 4, 26))

    assert summary["games"] == 5
    assert summary["hits"] == 5
    assert summary["at_bats"] == 10
    assert summary["batting_average"] == 0.5


def test_pitcher_game_log_parser_and_last_start_stats_include_strikeouts_and_walks():
    person = {
        "stats": [
            {
                "splits": [
                    {
                        "date": "2026-04-20",
                        "game": {"gamePk": 10},
                        "stat": {
                            "inningsPitched": "1.0",
                            "hits": "1",
                            "strikeOuts": "2",
                            "baseOnBalls": "0",
                            "gamesStarted": "0",
                            "gamesPitched": "1",
                        },
                    },
                    {
                        "date": "2026-04-21",
                        "game": {"gamePk": 11},
                        "stat": {
                            "inningsPitched": "6.0",
                            "hits": "3",
                            "strikeOuts": "8",
                            "baseOnBalls": "2",
                            "gamesStarted": "1",
                            "gamesPitched": "1",
                        },
                    },
                    {
                        "date": "2026-04-27",
                        "game": {"gamePk": 12},
                        "stat": {
                            "inningsPitched": "7.0",
                            "hits": "2",
                            "strikeOuts": "10",
                            "baseOnBalls": "1",
                            "gamesStarted": "1",
                            "gamesPitched": "1",
                        },
                    },
                ]
            }
        ]
    }

    entries = parse_pitcher_game_entries(person)
    last_start = pitcher_last_start_stats(entries, target_date=date(2026, 4, 26))

    assert entries[1].strikeouts == 8
    assert entries[1].walks == 2
    assert last_start == {
        "date": date(2026, 4, 21),
        "innings": 6.0,
        "hits": 3,
        "strikeouts": 8,
        "walks": 2,
    }


def test_good_candidate_is_pickable():
    result = evaluate_stop_valves(base_features())
    assert result.pickable
    assert result.hard_pass_reasons == ()


def test_h2h_pa_stop_valve_is_hard():
    result = evaluate_stop_valves(base_features(h2h_pa=1))
    assert not result.pickable
    assert "H2H PA below 2" in result.hard_pass_reasons


def test_pitcher_stuff_plus_stop_valve_is_hard():
    result = evaluate_stop_valves(base_features(pitcher_stuff_plus=99.0))
    assert not result.pickable
    assert "Starter Stuff+ over 95" in result.hard_pass_reasons


def test_dominant_start_stop_valve_requires_all_criteria():
    result = evaluate_stop_valves(
        base_features(
            pitcher_last_start_ip=6.0,
            pitcher_last_start_hits=3,
            pitcher_last_start_strikeouts=8,
            pitcher_last_start_walks=2,
        )
    )

    assert not result.pickable
    assert "Starter's most recent start was dominant (6+ IP, 8+ K, <=3 H, <=2 BB)" in result.hard_pass_reasons


def test_dominant_start_stop_valve_does_not_trigger_on_near_miss():
    result = evaluate_stop_valves(
        base_features(
            pitcher_last_start_ip=6.0,
            pitcher_last_start_hits=4,
            pitcher_last_start_strikeouts=8,
            pitcher_last_start_walks=2,
        )
    )

    assert result.pickable
    assert "Starter's most recent start was dominant (6+ IP, 8+ K, <=3 H, <=2 BB)" not in result.hard_pass_reasons


def test_candidate_row_includes_pitcher_last_start_fields():
    row = scored_candidate_to_row(score_candidate(base_features()))

    assert row["pitcher_last_start_date"] == "2026-04-20"
    assert row["pitcher_last_start_ip"] == "5.2"
    assert row["pitcher_last_start_hits"] == "5"
    assert row["pitcher_last_start_strikeouts"] == "7"
    assert row["pitcher_last_start_walks"] == "1"


def test_hitter_pa_per_game_stop_valve_is_hard():
    result = evaluate_stop_valves(base_features(hitter_pa_per_game_season=4.1))
    assert not result.pickable
    assert "Hitter season PA/G under 4.2" in result.hard_pass_reasons


def test_hitter_discipline_stop_valves_are_hard():
    result = evaluate_stop_valves(base_features(hitter_bb_rate_500_pa=0.130, hitter_k_rate_season=0.230))
    assert not result.pickable
    assert "Hitter last-500 PA BB rate over 12%" in result.hard_pass_reasons
    assert "Hitter season K rate over 22%" in result.hard_pass_reasons


def test_both_hands_stop_valve_allows_different_qualified_windows_by_hand():
    result = evaluate_stop_valves(
        base_features(
            hitter_split_ba_season_vs_lhp=0.300,
            hitter_split_pa_season_vs_lhp=50,
            hitter_split_ba_500_vs_lhp=0.260,
            hitter_split_ba_1500_vs_lhp=0.260,
            hitter_split_ba_season_vs_rhp=0.250,
            hitter_split_pa_season_vs_rhp=70,
            hitter_split_ba_500_vs_rhp=0.270,
            hitter_split_ba_1500_vs_rhp=0.260,
        )
    )

    assert result.pickable
    assert "Hitter is not at least .265 against both pitcher hands in season/500/1500 windows" not in result.hard_pass_reasons


def test_both_hands_stop_valve_ignores_low_sample_current_season_split():
    result = evaluate_stop_valves(
        base_features(
            hitter_split_ba_season_vs_lhp=0.400,
            hitter_split_pa_season_vs_lhp=49,
            hitter_split_ba_season_vs_rhp=0.400,
            hitter_split_pa_season_vs_rhp=49,
            hitter_split_ba_500_vs_lhp=0.260,
            hitter_split_ba_1500_vs_lhp=0.260,
            hitter_split_ba_500_vs_rhp=0.260,
            hitter_split_ba_1500_vs_rhp=0.260,
        )
    )

    assert not result.pickable
    assert "Hitter is not at least .265 against both pitcher hands in season/500/1500 windows" in result.hard_pass_reasons


def test_matchup_hand_stop_valve_still_uses_270_from_500_or_1500_windows():
    result = evaluate_stop_valves(
        base_features(
            pitcher_hand="R",
            hitter_split_ba_season_vs_lhp=0.300,
            hitter_split_pa_season_vs_lhp=60,
            hitter_split_ba_season_vs_rhp=0.300,
            hitter_split_pa_season_vs_rhp=60,
            hitter_matchup_hand_ba_500=0.260,
            hitter_matchup_hand_ba_1500=0.260,
        )
    )

    assert not result.pickable
    assert "Hitter is not at least .270 against today's pitcher hand" in result.hard_pass_reasons


def test_score_candidate_preserves_score_even_when_not_pickable():
    result = score_candidate(base_features(hitter_ba_25_ab=0.120))
    assert result.score > 0
    assert not result.valve_result.pickable
    assert "Hitter BA under .150 over last 25 AB" in result.valve_result.hard_pass_reasons


def test_h2h_uses_career_matchup_frame_without_polluting_pitcher_splits():
    recent_pitcher_rows = pd.DataFrame(
        [
            {
                "game_date": "2025-06-01",
                "game_pk": 1,
                "at_bat_number": 1,
                "pitch_number": 1,
                "batter": 10,
                "pitcher": 20,
                "stand": "R",
                "p_throws": "R",
                "events": "single",
                "description": "hit_into_play",
                "launch_speed": 95.0,
                "estimated_ba_using_speedangle": 0.600,
            }
        ]
    )
    career_h2h_rows = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "game_date": "2021-06-01",
                        "game_pk": 2,
                        "at_bat_number": 1,
                        "pitch_number": 1,
                        "batter": 10,
                        "pitcher": 20,
                        "stand": "R",
                        "p_throws": "R",
                        "events": "strikeout",
                        "description": "swinging_strike",
                        "launch_speed": None,
                        "estimated_ba_using_speedangle": None,
                    }
                ]
            ),
            recent_pitcher_rows,
        ],
        ignore_index=True,
    )

    store = StatcastFeatureStore(pd.DataFrame(), recent_pitcher_rows, h2h_df=career_h2h_rows)

    h2h = store.h2h(10, 20)
    assert h2h.pa == 2
    assert h2h.hits == 1
    assert h2h.k_rate == 0.5
    assert h2h.hit_rate == 0.5
    assert store.pitcher_split_opp_ba(20, "R") == 1.0


def test_hitter_split_windows_include_pregame_current_season_pa_by_hand():
    batter_rows = pd.DataFrame(
        [
            {
                "game_date": "2026-06-10",
                "game_pk": 1,
                "at_bat_number": 1,
                "pitch_number": 1,
                "batter": 10,
                "pitcher": 20,
                "stand": "L",
                "p_throws": "L",
                "events": "single",
                "description": "hit_into_play",
            },
            {
                "game_date": "2026-06-11",
                "game_pk": 2,
                "at_bat_number": 1,
                "pitch_number": 1,
                "batter": 10,
                "pitcher": 21,
                "stand": "L",
                "p_throws": "L",
                "events": "walk",
                "description": "ball",
            },
            {
                "game_date": "2026-06-12",
                "game_pk": 3,
                "at_bat_number": 1,
                "pitch_number": 1,
                "batter": 10,
                "pitcher": 22,
                "stand": "L",
                "p_throws": "L",
                "events": "field_out",
                "description": "hit_into_play",
            },
            {
                "game_date": "2026-06-13",
                "game_pk": 4,
                "at_bat_number": 1,
                "pitch_number": 1,
                "batter": 10,
                "pitcher": 23,
                "stand": "L",
                "p_throws": "R",
                "events": "single",
                "description": "hit_into_play",
            },
            {
                "game_date": "2026-06-15",
                "game_pk": 5,
                "at_bat_number": 1,
                "pitch_number": 1,
                "batter": 10,
                "pitcher": 24,
                "stand": "L",
                "p_throws": "L",
                "events": "home_run",
                "description": "hit_into_play",
            },
        ]
    )
    batter_rows["launch_speed"] = None
    batter_rows["estimated_ba_using_speedangle"] = None

    store = StatcastFeatureStore(batter_rows, pd.DataFrame())

    splits = store.hitter_split_windows(10, target_date=date(2026, 6, 15))

    assert splits["pa_season_vs_lhp"] == 3
    assert splits["ba_season_vs_lhp"] == 0.5
    assert splits["pa_season_vs_rhp"] == 1
    assert splits["ba_season_vs_rhp"] == 1.0


def test_candidate_upsert_preserves_existing_result_columns():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidates.csv"
        row = {
            "date": "2026-04-26",
            "player": "Test Hitter",
            "player_id": "1",
            "team": "BOS",
            "opponent": "NYY",
            "game_pk": "100",
            "score": "60.00",
        }
        upsert_candidate_rows(path, [row])
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["result_hit"] = "1"
        rows[0]["result_hits"] = "2"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        upsert_candidate_rows(path, [{**row, "score": "61.00"}])
        with path.open(newline="") as f:
            updated = list(csv.DictReader(f))[0]
        assert updated["score"] == "61.00"
        assert updated["result_hit"] == "1"
        assert updated["result_hits"] == "2"


def test_pending_result_status_is_not_treated_as_graded():
    assert not has_result_data({"result_status": "Scheduled"})
    assert has_result_data({"result_status": "Final", "result_hits": "0"})


def test_result_status_values_are_machine_readable():
    assert result_status_for_game_status("Postponed") == RESULT_STATUS_POSTPONED
    assert result_status_for_game_status("Scheduled") == RESULT_STATUS_PENDING
    assert normalize_row_result_status({"result_status": "Game Over", "result_hit": "0"}) == RESULT_STATUS_FINAL
    assert (
        normalize_row_result_status({"result_status": "Final - no appearance", "result_hit": ""})
        == RESULT_STATUS_NO_APPEARANCE
    )


def test_results_updater_marks_postponed_rows_without_grading_them():
    original_client = results_module.MLBClient

    class FakeClient:
        def schedule(self, start, end, hydrate=""):
            return [
                {
                    "gamePk": 100,
                    "status": {
                        "abstractGameState": "Final",
                        "detailedState": "Postponed",
                    },
                    "teams": {
                        "away": {"team": {"id": 111}},
                        "home": {"team": {"id": 147}},
                    },
                }
            ]

    results_module.MLBClient = FakeClient
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["date", "player", "player_id", "team", "opponent", "game_pk", "score"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "date": "2026-04-26",
                        "player": "Test Hitter",
                        "player_id": "1",
                        "team": "BOS",
                        "opponent": "NYY",
                        "game_pk": "100",
                        "score": "60.00",
                    }
                )

            summary = update_results_csv(path)
            with path.open(newline="") as f:
                row = next(csv.DictReader(f))
    finally:
        results_module.MLBClient = original_client

    assert summary["updated"] == 1
    assert summary["pending"] == 0
    assert summary["postponed"] == 1
    assert row["result_status"] == RESULT_STATUS_POSTPONED
    assert row["result_hit"] == ""
    assert row["result_hits"] == ""


def test_candidate_upsert_replaces_ungraded_rows_for_same_date():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidates.csv"
        old_row = {
            "date": "2026-04-26",
            "player": "Old Hitter",
            "player_id": "1",
            "team": "BOS",
            "opponent": "NYY",
            "game_pk": "100",
            "score": "50.00",
        }
        result_row = {
            "date": "2026-04-26",
            "player": "Graded Hitter",
            "player_id": "2",
            "team": "BOS",
            "opponent": "NYY",
            "game_pk": "101",
            "score": "51.00",
            "result_hit": "1",
        }
        new_row = {
            "date": "2026-04-26",
            "player": "New Hitter",
            "player_id": "3",
            "team": "BOS",
            "opponent": "NYY",
            "game_pk": "102",
            "score": "60.00",
        }
        upsert_candidate_rows(path, [old_row, result_row])
        upsert_candidate_rows(path, [new_row], replace_date="2026-04-26")
        with path.open(newline="") as f:
            names = {row["player"] for row in csv.DictReader(f)}
        assert "Old Hitter" not in names
        assert "Graded Hitter" in names
        assert "New Hitter" in names


def test_game_context_hydrates_venue_coordinates():
    class FakeClient:
        def schedule(self, start, end):
            return [
                {
                    "gamePk": 123,
                    "gameDate": "2026-04-26T20:05:00Z",
                    "teams": {
                        "away": {"team": {"id": 135}},
                        "home": {"team": {"id": 109}},
                    },
                    "venue": {"id": 2, "name": "Oriole Park at Camden Yards"},
                }
            ]

        def venue(self, venue_id):
            assert venue_id == 2
            return {
                "venues": [
                    {
                        "location": {
                            "defaultCoordinates": {
                                "latitude": 39.283787,
                                "longitude": -76.621689,
                            }
                        }
                    }
                ]
            }

    games = get_games_for_date(FakeClient(), date(2026, 4, 26))
    assert len(games) == 1
    assert games[0].venue_latitude == 39.283787
    assert games[0].venue_longitude == -76.621689


def test_game_context_uses_coordinate_override_for_mexico_city_venue():
    class FakeClient:
        def schedule(self, start, end):
            return [
                {
                    "gamePk": 124,
                    "gameDate": "2026-04-26T20:05:00Z",
                    "teams": {
                        "away": {"team": {"id": 135}},
                        "home": {"team": {"id": 109}},
                    },
                    "venue": {"id": 5340, "name": "Estadio Alfredo Harp Helu"},
                }
            ]

        def venue(self, venue_id):
            return {"venues": [{"location": {"city": "Mexico City", "country": "Mexico"}}]}

    games = get_games_for_date(FakeClient(), date(2026, 4, 26))
    assert len(games) == 1
    assert games[0].venue_latitude == 19.403794
    assert games[0].venue_longitude == -99.085594


def test_weather_probability_uses_max_during_game_window():
    original_get = weather.requests.get

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "hourly": {
                    "time": [
                        "2026-04-26T20:00",
                        "2026-04-26T21:00",
                        "2026-04-26T22:00",
                        "2026-04-26T23:00",
                        "2026-04-27T00:00",
                    ],
                    "precipitation_probability": [5, 12, 55, 30, 0],
                    "temperature_2m": [70.0, 71.2, 72.4, 73.1, 72.0],
                }
            }

    def fake_get(url, params, timeout):
        assert params["start_date"] == "2026-04-26"
        assert params["end_date"] == "2026-04-27"
        return FakeResponse()

    weather.requests.get = fake_get
    try:
        probability, warning = weather.fetch_precipitation_probability(
            latitude=39.283787,
            longitude=-76.621689,
            game_datetime_utc=datetime(2026, 4, 26, 20, 5, tzinfo=timezone.utc),
        )
        forecast, forecast_warning = weather.fetch_weather_forecast(
            latitude=39.283787,
            longitude=-76.621689,
            game_datetime_utc=datetime(2026, 4, 26, 20, 5, tzinfo=timezone.utc),
        )
    finally:
        weather.requests.get = original_get

    assert probability == 55
    assert warning is None
    assert forecast.precipitation_probability == 55
    assert forecast.temperature_f == 70.0
    assert forecast_warning is None


def test_weather_updater_fills_precipitation_column():
    original_client = weather_update.MLBClient
    original_get_games = weather_update.get_games_for_date
    original_fetch = weather_update.fetch_weather_forecast

    class FakeClient:
        pass

    def fake_get_games(client, day):
        return [
            SimpleNamespace(
                game_pk=100,
                venue_latitude=39.283787,
                venue_longitude=-76.621689,
                game_datetime_utc=datetime(2026, 4, 26, 20, 5, tzinfo=timezone.utc),
                away_abbr="BOS",
                home_abbr="BAL",
            )
        ]

    def fake_fetch(*, latitude, longitude, game_datetime_utc):
        return weather.WeatherForecast(precipitation_probability=42.0, temperature_f=71.5), None

    weather_update.MLBClient = FakeClient
    weather_update.get_games_for_date = fake_get_games
    weather_update.fetch_weather_forecast = fake_fetch
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["date", "player", "player_id", "game_pk", "score"])
                writer.writeheader()
                writer.writerow(
                    {
                        "date": "2026-04-26",
                        "player": "Test Hitter",
                        "player_id": "1",
                        "game_pk": "100",
                        "score": "60.00",
                    }
                )
            summary = weather_update.update_weather_csv(path)
            with path.open(newline="") as f:
                row = next(csv.DictReader(f))
    finally:
        weather_update.MLBClient = original_client
        weather_update.get_games_for_date = original_get_games
        weather_update.fetch_weather_forecast = original_fetch

    assert summary["updated"] == 1
    assert row["precip_probability"] == "42.0"
    assert row["forecast_temperature_f"] == "71.5"


def test_bullpen_updater_fills_relief_batting_average():
    original_client = bullpen_update.MLBClient
    original_season_start = bullpen_update.season_start_for
    original_compute = bullpen_update.compute_bullpen_stats

    class FakeClient:
        pass

    bullpen_update.MLBClient = FakeClient
    bullpen_update.season_start_for = lambda client, year: date(2026, 3, 20)
    bullpen_update.compute_bullpen_stats = lambda client, season_start, end, season: {
        147: {"hits_per_inning": 1.05, "opponent_batting_average": 0.285}
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["date", "player", "opponent_id", "bullpen_hpi"])
                writer.writeheader()
                writer.writerow(
                    {
                        "date": "2026-04-26",
                        "player": "Test Hitter",
                        "opponent_id": "147",
                        "bullpen_hpi": "",
                    }
                )
            summary = bullpen_update.update_bullpen_csv(path)
            with path.open(newline="") as f:
                row = next(csv.DictReader(f))
    finally:
        bullpen_update.MLBClient = original_client
        bullpen_update.season_start_for = original_season_start
        bullpen_update.compute_bullpen_stats = original_compute

    assert summary["updated"] == 1
    assert row["bullpen_opp_ba"] == "0.285"
    assert row["bullpen_hpi"] == "1.050"


def test_web_candidate_payload_uses_weighted_bucket_points():
    row = {
        "date": "2026-04-27",
        "player": "Test Hitter",
        "player_id": "1",
        "team": "BOS",
        "opponent": "NYY",
        "game_pk": "100",
        "pickable": "Y",
        "score": "25.00",
        "result_status": "Game Over",
        "hitter_last_5_games_played": "5",
        "hitter_last_5_games_hits": "5",
        "hitter_last_5_games_ab": "10",
        "hitter_last_5_games_ba": "0.500",
        "h2h_pa": "10",
        "h2h_hits": "5",
        "precip_probability": "12.0",
        "forecast_temperature_f": "72.4",
        "park_hit_factor": "110.0",
        "component_hitter.hipa_2500_pa": "100.00",
        "component_hitter.pa_per_game_season": "100.00",
        "component_hitter.hipa_500_pa": "100.00",
        "component_hitter.hipa_75_ab": "100.00",
    }
    payload = candidate_payload(
        row,
        1,
        {
            (100, 1): {
                "state": "hit",
                "status": "In Progress",
                "hits": 1,
                "venue_name": "Fenway Park",
                "game_start_time_utc": "2026-04-27T23:10:00Z",
            }
        },
    )
    context_labels = [item["label"] for item in payload["factors"]["context"]]
    assert payload["buckets"]["hitter"]["points"] == 25.0
    assert sum(bucket["points"] for bucket in payload["buckets"].values()) == 25.0
    assert payload["game_state"] == "hit"
    assert payload["game_state_label"] == "Hit recorded"
    assert payload["result_status"] == RESULT_STATUS_FINAL
    assert payload["game_hits"] == 1
    assert payload["precip_probability"] == "12.0"
    assert payload["forecast_temperature_f"] == "72.4"
    assert payload["venue_name"] == "Fenway Park"
    assert payload["roofed_ballpark"] is False
    assert payload["weather_label"] == "Rain: 12.0%"
    assert payload["game_start_time_utc"] == "2026-04-27T23:10:00Z"
    assert payload["hot_streak"] is True
    assert payload["hot_streak_tooltip"] == "5-10"
    assert payload["h2h_record"] == "5-10"
    assert "Rain" not in context_labels
    assert "Park BA" not in context_labels


def test_roofed_ballpark_payload_uses_dome_weather_label():
    row = {
        "date": "2026-04-27",
        "player": "Test Hitter",
        "player_id": "1",
        "team": "ARI",
        "opponent": "LAD",
        "game_pk": "100",
        "pickable": "Y",
        "score": "25.00",
        "venue_name": "Chase Field",
        "precip_probability": "88.0",
        "forecast_temperature_f": "97.8",
    }

    payload = candidate_payload(row, 1)

    assert is_roofed_ballpark("Chase Field") is True
    assert is_roofed_ballpark("Daikin Park") is True
    assert payload["roofed_ballpark"] is True
    assert payload["weather_label"] == "Dome"
    assert payload["precip_probability"] == "88.0"


def test_compute_hitter_windows_includes_current_season_ba():
    entries = [
        HitterPlayEntry(date(2025, 9, 1), 1, 1, is_hit=1, is_at_bat=1),
        HitterPlayEntry(date(2026, 4, 1), 2, 1, is_hit=1, is_at_bat=1),
        HitterPlayEntry(date(2026, 4, 2), 3, 1, is_hit=0, is_at_bat=1),
        HitterPlayEntry(date(2026, 4, 3), 4, 1, is_hit=1, is_at_bat=1),
        HitterPlayEntry(date(2026, 4, 4), 5, 1, is_hit=1, is_at_bat=0),
    ]

    windows = compute_hitter_windows(entries, target_date=date(2026, 4, 5))

    assert windows["ba_season"] == 2 / 3
    assert windows["ba_season_hits"] == 2
    assert windows["ba_season_sample"] == 3


def test_web_candidate_payload_requires_five_games_for_hot_streak():
    row = {
        "date": "2026-04-27",
        "player": "Short Sample",
        "player_id": "1",
        "team": "BOS",
        "opponent": "NYY",
        "game_pk": "100",
        "pickable": "Y",
        "score": "25.00",
        "hitter_last_5_games_played": "1",
        "hitter_last_5_games_hits": "4",
        "hitter_last_5_games_ab": "4",
        "hitter_last_5_games_ba": "1.000",
    }

    payload = candidate_payload(row, 1)

    assert payload["hot_streak"] is False
    assert payload["hot_streak_tooltip"] == ""


def test_web_game_phase_prioritizes_postponed_over_final():
    game = {
        "status": {
            "abstractGameState": "Final",
            "detailedState": "Postponed",
        }
    }
    assert export_web.game_phase(game) == "postponed"


def test_web_export_includes_congregation_rows_beyond_limit():
    original_state_lookup = export_web.build_game_state_lookup
    export_web.build_game_state_lookup = lambda target_day, rows: {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidates = tmp_path / "candidates.csv"
            fieldnames = ["date", "player", "player_id", "team", "opponent", "game_pk", "pickable", "score", "h2h_pa", "h2h_hits"]
            with candidates.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for idx in range(1, 13):
                    writer.writerow(
                        {
                            "date": "2026-04-28",
                            "player": f"Hitter {idx}",
                            "player_id": str(idx),
                            "team": "BOS",
                            "opponent": "NYY",
                            "game_pk": str(100 + idx),
                            "pickable": "N",
                            "score": f"{100 - idx:.2f}",
                            "h2h_pa": str(idx),
                            "h2h_hits": "1",
                        }
                    )
            congregation = tmp_path / "congregation.csv"
            congregation.write_text("player_id,player,status,aliases\n12,Hitter 12,Publisher,\n", encoding="utf-8")
            out_json = tmp_path / "top_picks.json"
            payload = export_web.export_web_payload(
                candidates_csv=candidates,
                out_json=out_json,
                index_json=tmp_path / "dashboard_index.json",
                archive_dir=tmp_path / "dashboards",
                congregation_csv=congregation,
                target_date="2026-04-28",
                limit=3,
                archive=False,
            )
    finally:
        export_web.build_game_state_lookup = original_state_lookup

    assert [pick["rank"] for pick in payload["picks"]] == [1, 2, 3, 12]
    assert payload["picks"][-1]["player"] == "Hitter 12"
    assert payload["picks"][-1]["congregation_status"] == "Publisher"


def test_web_export_archives_dated_dashboards_and_index():
    original_state_lookup = export_web.build_game_state_lookup
    export_web.build_game_state_lookup = lambda target_day, rows: {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidates = tmp_path / "candidates.csv"
            with candidates.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "date",
                        "player",
                        "player_id",
                        "team",
                        "opponent",
                        "game_pk",
                        "pickable",
                        "score",
                        "h2h_pa",
                        "h2h_hits",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "date": "2026-04-27",
                        "player": "Yesterday Hitter",
                        "player_id": "1",
                        "team": "BOS",
                        "opponent": "NYY",
                        "game_pk": "100",
                        "pickable": "N",
                        "score": "51.00",
                        "h2h_pa": "3",
                        "h2h_hits": "1",
                    }
                )
                writer.writerow(
                    {
                        "date": "2026-04-28",
                        "player": "Today Hitter",
                        "player_id": "2",
                        "team": "BOS",
                        "opponent": "NYY",
                        "game_pk": "101",
                        "pickable": "N",
                        "score": "61.00",
                        "h2h_pa": "4",
                        "h2h_hits": "2",
                    }
                )
            out_json = tmp_path / "data" / "top_picks.json"
            index_json = tmp_path / "data" / "dashboard_index.json"
            archive_dir = tmp_path / "data" / "dashboards"
            export_web.export_web_payload(
                candidates_csv=candidates,
                out_json=out_json,
                index_json=index_json,
                archive_dir=archive_dir,
                target_date="2026-04-27",
            )
            export_web.export_web_payload(
                candidates_csv=candidates,
                out_json=out_json,
                index_json=index_json,
                archive_dir=archive_dir,
                target_date="2026-04-28",
            )
            index = json.loads(index_json.read_text())
            archived_27 = (archive_dir / "2026-04-27.json").exists()
            archived_28 = (archive_dir / "2026-04-28.json").exists()
    finally:
        export_web.build_game_state_lookup = original_state_lookup

    assert archived_27
    assert archived_28
    assert index["active_date"] == "2026-04-28"
    assert [row["date"] for row in index["dashboards"]] == ["2026-04-28", "2026-04-27"]
