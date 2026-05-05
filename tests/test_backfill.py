from datetime import date

from statbirt.backfill import historical_lineups_for_date, regular_season_dates


class FakeClient:
    def schedule(self, start, end, *, hydrate=""):
        return [
            {"gameType": "S", "officialDate": "2026-03-24"},
            {"gameType": "R", "officialDate": "2026-03-25", "gamePk": 1},
            {"gameType": "R", "officialDate": "2026-03-25", "gamePk": 2},
            {"gameType": "R", "officialDate": "2026-03-26", "gamePk": 3},
            {"gameType": "A", "officialDate": "2026-07-14"},
        ]

    def boxscore(self, game_pk):
        return {
            "teams": {
                "away": {
                    "team": {"id": 111},
                    "battingOrder": [1, 2, 3],
                    "players": {
                        "ID1": {"person": {"fullName": "First Hitter"}},
                        "ID2": {"person": {"fullName": "Second Hitter"}},
                        "ID3": {"person": {"fullName": "Third Hitter"}},
                    },
                },
                "home": {
                    "team": {"id": 147},
                    "battingOrder": [4],
                    "players": {"ID4": {"person": {"fullName": "Home Hitter"}}},
                },
            }
        }


def test_regular_season_dates_are_unique_and_regular_only():
    assert regular_season_dates(FakeClient(), date(2026, 3, 1), date(2026, 7, 31)) == [
        date(2026, 3, 25),
        date(2026, 3, 26),
    ]


def test_historical_lineups_use_boxscore_batting_order():
    lineups = historical_lineups_for_date(FakeClient(), date(2026, 3, 25))
    assert lineups[111]["confirmed"] is True
    assert lineups[111]["players_by_id"] == {1: 1, 2: 2, 3: 3}
    assert lineups[111]["players_by_name"]["second hitter"] == 2
    assert lineups[147]["players_by_id"] == {4: 1}
