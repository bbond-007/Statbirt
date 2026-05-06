from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class GameContext:
    game_pk: int | None
    game_date: date
    game_datetime_utc: datetime | None
    away_id: int
    home_id: int
    away_abbr: str
    home_abbr: str
    away_probable_pitcher_id: int | None
    away_probable_pitcher_name: str
    home_probable_pitcher_id: int | None
    home_probable_pitcher_name: str
    venue_id: int | None = None
    venue_name: str = ""
    venue_latitude: float | None = None
    venue_longitude: float | None = None


@dataclass
class CandidateFeatures:
    target_date: date
    player_id: int
    player_name: str
    team_id: int
    team: str
    opponent_id: int
    opponent: str
    game_pk: int | None
    is_home: bool
    game_start_time_utc: datetime | None = None
    venue_name: str = ""
    confirmed_lineup: bool = False
    lineup_slot: float | None = None
    starts_last_5: int = 0
    hitter_last_5_games_hits: int | None = None
    hitter_last_5_games_ab: int | None = None
    hitter_last_5_games_ba: float | None = None
    pitcher_id: int | None = None
    pitcher_name: str = "TBD"
    pitcher_hand: str = "?"
    batter_stand: str = "?"
    same_division: bool = False
    doubleheader: bool = False
    precipitation_probability: float | None = None
    forecast_temperature_f: float | None = None
    opener_risk: bool = False
    hitter_hipa_2500_pa: float | None = None
    hitter_pa_per_game_season: float | None = None
    hitter_ba_2500_ab: float | None = None
    hitter_hipa_500_pa: float | None = None
    hitter_hipa_75_ab: float | None = None
    hitter_ba_75_ab: float | None = None
    hitter_ba_25_ab: float | None = None
    hitter_ba_500_ab: float | None = None
    hitter_bb_rate_season: float | None = None
    hitter_bb_rate_500_pa: float | None = None
    hitter_whiff_rate_season: float | None = None
    hitter_whiff_rate_500_pa: float | None = None
    hitter_k_rate_season: float | None = None
    hitter_k_rate_500_pa: float | None = None
    hitter_split_ba_500_vs_lhp: float | None = None
    hitter_split_ba_500_vs_rhp: float | None = None
    hitter_split_ba_1500_vs_lhp: float | None = None
    hitter_split_ba_1500_vs_rhp: float | None = None
    hitter_matchup_hand_ba_500: float | None = None
    hitter_matchup_hand_ba_1500: float | None = None
    pitcher_hpi_350: float | None = None
    pitcher_hpi_200: float | None = None
    pitcher_hpi_season: float | None = None
    pitcher_hits_last_18_ip: int | None = None
    pitcher_stuff_plus: float | None = None
    h2h_pa: int = 0
    h2h_hit_rate: float | None = None
    h2h_whiff_rate: float | None = None
    h2h_k_rate: float | None = None
    h2h_exit_velocity: float | None = None
    h2h_xba: float | None = None
    pitcher_lr_opp_ba: float | None = None
    pitcher_lr_opp_ba_50: float | None = None
    pitcher_lr_opp_ba_200: float | None = None
    inferred_pitch_type_ba: float | None = None
    inferred_pitch_type_xba: float | None = None
    inferred_pitch_type_coverage: float | None = None
    bullpen_hpi: float | None = None
    bullpen_opp_ba: float | None = None
    sprint_speed: float | None = None
    park_hit_factor: float | None = None
    expected_pa: float | None = None
    missing_data: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    weight: float
    raw_value: float | str | None
    subscore: float
    points: float


@dataclass(frozen=True)
class ValveResult:
    pickable: bool
    hard_pass_reasons: tuple[str, ...]
    concerns: tuple[str, ...]


@dataclass(frozen=True)
class ScoredCandidate:
    features: CandidateFeatures
    score: float
    components: tuple[ScoreComponent, ...]
    valve_result: ValveResult
