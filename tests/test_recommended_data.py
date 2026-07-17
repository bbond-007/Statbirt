from datetime import date
import json

import pandas as pd

from statbirt import mlb_api
from statbirt.mlb_api import compute_bullpen_stats, load_roster_availability
from statbirt.savant import StatcastFeatureStore
from statbirt.savant_snapshots import load_bat_tracking_snapshot, load_oaa_snapshot


def statcast_rows():
    base = {
        "batter": 10,
        "pitcher": 20,
        "stand": "R",
        "p_throws": "R",
        "pitch_type": "FF",
        "description": "hit_into_play",
        "season_window": 2026,
        "season_weight": 1.0,
        "bat_speed": 72.0,
        "swing_length": 7.2,
        "release_speed": 95.0,
        "pfx_x": -0.5,
        "pfx_z": 1.2,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "game_date": "2026-06-01",
                "game_pk": 1,
                "at_bat_number": 1,
                "pitch_number": 1,
                "events": "single",
                "launch_speed": 100.0,
                "launch_angle": 15.0,
                "estimated_ba_using_speedangle": 0.8,
            },
            {
                **base,
                "game_date": "2026-06-02",
                "game_pk": 2,
                "at_bat_number": 1,
                "pitch_number": 1,
                "events": "field_out",
                "launch_speed": 90.0,
                "launch_angle": 40.0,
                "estimated_ba_using_speedangle": 0.2,
            },
            {
                **base,
                "game_date": "2026-06-03",
                "game_pk": 3,
                "at_bat_number": 1,
                "pitch_number": 1,
                "events": "strikeout",
                "description": "swinging_strike",
                "launch_speed": None,
                "launch_angle": None,
                "estimated_ba_using_speedangle": None,
            },
        ]
    )


def test_contact_quality_uses_strikeout_xba_denominator_and_correct_ev50_direction():
    rows = statcast_rows()
    store = StatcastFeatureStore(rows, rows)

    hitter = store.hitter_contact_quality(10, target_date=date(2026, 7, 1))
    pitcher = store.pitcher_contact_quality_allowed(20, target_date=date(2026, 7, 1))

    assert round(hitter["xba"], 3) == 0.333
    assert hitter["xba_denominator"] == 3
    assert hitter["hard_hit_rate"] == 0.5
    assert hitter["sweet_spot_rate"] == 0.5
    assert hitter["ev50"] == 100.0
    assert pitcher["ev50"] == 90.0


def test_prospective_savant_snapshots_parse_bat_tracking_and_oaa(tmp_path):
    target = date(2026, 7, 17)
    bat_dir = tmp_path / "bat_tracking"
    oaa_dir = tmp_path / "outs_above_average"
    bat_dir.mkdir(parents=True)
    oaa_dir.mkdir(parents=True)
    (bat_dir / f"{target}.csv").write_text(
        "id,name,swings_competitive,contact,avg_bat_speed,swing_length,squared_up_per_bat_contact,blast_per_bat_contact\n"
        "10,Test Hitter,100,82,74.2,7.1,0.31,0.19\n",
        encoding="utf-8",
    )
    (oaa_dir / f"{target}.csv").write_text(
        "player_id,display_team_name,primary_pos_formatted,outs_above_average\n"
        "20,Red Sox,SS,4\n21,Red Sox,CF,3\n",
        encoding="utf-8",
    )
    metadata = {
        "snapshot_id": "test-snapshot",
        "source_hash": "abc",
        "source_max_game_date": "2026-07-16",
    }
    (bat_dir / f"{target}.json").write_text(json.dumps(metadata), encoding="utf-8")
    (oaa_dir / f"{target}.json").write_text(json.dumps(metadata), encoding="utf-8")

    bat, _ = load_bat_tracking_snapshot(target, snapshot_dir=tmp_path)
    oaa, _ = load_oaa_snapshot(
        target,
        {111: {"name": "Boston Red Sox", "abbr": "BOS"}},
        snapshot_dir=tmp_path,
    )

    assert bat[10]["competitive_contact_rate"] == 0.82
    assert bat[10]["squared_up_per_contact"] == 0.31
    assert oaa[111]["team_oaa"] == 7.0
    assert oaa[111]["infield_oaa"] == 4.0
    assert oaa[111]["outfield_oaa"] == 3.0


def test_oaa_team_aliases_include_dbacks(tmp_path):
    target = date(2026, 7, 17)
    oaa_dir = tmp_path / "outs_above_average"
    oaa_dir.mkdir(parents=True)
    (oaa_dir / f"{target}.csv").write_text(
        "player_id,display_team_name,primary_pos_formatted,outs_above_average\n"
        "20,D-backs,SS,2\n",
        encoding="utf-8",
    )
    (oaa_dir / f"{target}.json").write_text(
        json.dumps({"snapshot_id": "test", "source_hash": "abc"}), encoding="utf-8"
    )

    oaa, _ = load_oaa_snapshot(
        target,
        {109: {"name": "Arizona Diamondbacks", "abbr": "AZ"}},
        snapshot_dir=tmp_path,
    )

    assert oaa[109]["team_oaa"] == 2.0


def test_bullpen_workload_uses_only_prior_three_days(monkeypatch, tmp_path):
    class Client:
        def schedule(self, start, end, hydrate=""):
            return [{"gamePk": 1, "officialDate": "2026-07-16", "status": {}}]

        def boxscore(self, game_pk):
            return {
                "teams": {
                    "away": {
                        "team": {"id": 111},
                        "pitchers": [1, 2, 3],
                        "players": {
                            "ID2": {"stats": {"pitching": {"inningsPitched": "1.0", "hits": 1, "atBats": 4, "numberOfPitches": 18}}},
                            "ID3": {"stats": {"pitching": {"inningsPitched": "1.0", "hits": 0, "atBats": 3, "numberOfPitches": 16}}},
                        },
                    },
                    "home": {"team": {"id": 222}, "pitchers": [], "players": {}},
                }
            }

    monkeypatch.setattr(mlb_api, "cache_path", lambda *parts: tmp_path / "bullpen.pkl")
    monkeypatch.setattr(mlb_api, "load_pickle", lambda path: None)
    monkeypatch.setattr(mlb_api, "save_pickle", lambda path, value: None)
    result = compute_bullpen_stats(Client(), date(2026, 3, 1), date(2026, 7, 17), 2026)

    assert result[111]["pitches_last_3_days"] == 34
    assert result[111]["relief_appearances_last_3_days"] == 2
    assert result[111]["relievers_used_last_3_days"] == 2


def test_roster_availability_is_dated_and_tracks_recent_activation(monkeypatch, tmp_path):
    class Client:
        def roster(self, team_id, roster_type="active", season=None, roster_date=None):
            assert roster_date == date(2026, 7, 17)
            return [{"person": {"id": 10}, "status": {"code": "A"}}]

        def transactions(self, team_id, start, end):
            return [
                {
                    "person": {"id": 10},
                    "effectiveDate": "2026-07-14",
                    "typeCode": "SC",
                    "description": "Test Hitter recalled from Triple-A.",
                }
            ]

    monkeypatch.setattr(mlb_api, "cache_path", lambda *parts: tmp_path / "roster.pkl")
    monkeypatch.setattr(mlb_api, "load_pickle", lambda path: None)
    monkeypatch.setattr(mlb_api, "save_pickle", lambda path, value: None)
    players, covered = load_roster_availability(Client(), {111}, target_date=date(2026, 7, 17))

    assert covered == {111}
    assert players[10]["active_roster"] is True
    assert players[10]["days_since_activation"] == 3
    assert players[10]["last_transaction_type_code"] == "SC"


def test_empty_roster_response_does_not_mark_a_team_as_covered(monkeypatch, tmp_path):
    class Client:
        def roster(self, team_id, roster_type="active", season=None, roster_date=None):
            return []

        def transactions(self, team_id, start, end):
            return []

    monkeypatch.setattr(mlb_api, "cache_path", lambda *parts: tmp_path / "empty-roster.pkl")
    monkeypatch.setattr(mlb_api, "load_pickle", lambda path: None)
    monkeypatch.setattr(mlb_api, "save_pickle", lambda path, value: None)

    players, covered = load_roster_availability(Client(), {111}, target_date=date(2026, 7, 17))

    assert players == {}
    assert covered == set()
