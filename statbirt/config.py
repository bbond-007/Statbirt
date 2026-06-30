from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MANUAL_DIR = DATA_DIR / "manual"
DEFAULT_OUTPUT_CSV = DATA_DIR / "statbirt_candidates.csv"
DEFAULT_STUFF_PLUS_CSV = MANUAL_DIR / "stuff_plus.csv"


@dataclass(frozen=True)
class HitterWeights:
    hipa_2500_pa: float = 10.0
    pa_per_game_season: float = 7.0
    hipa_500_pa: float = 5.0
    hipa_75_ab: float = 3.0


@dataclass(frozen=True)
class StartingPitcherWeights:
    hits_per_inning_350: float = 10.0
    hits_per_inning_season: float = 5.0
    stuff_plus: float = 10.0


@dataclass(frozen=True)
class H2HWeights:
    direct_matchup: float = 10.0
    pitcher_lr_split: float = 5.0
    inferred_pitch_type: float = 5.0


@dataclass(frozen=True)
class BullpenWeights:
    hits_per_inning_season: float = 15.0


@dataclass(frozen=True)
class OtherWeights:
    road_game: float = 3.0
    division_matchup: float = 3.0
    sprint_speed: float = 3.0
    park_hit_factor: float = 3.0
    lineup_opportunity: float = 3.0


@dataclass(frozen=True)
class ScoreWeights:
    hitter: HitterWeights = field(default_factory=HitterWeights)
    starting_pitcher: StartingPitcherWeights = field(default_factory=StartingPitcherWeights)
    h2h: H2HWeights = field(default_factory=H2HWeights)
    bullpen: BullpenWeights = field(default_factory=BullpenWeights)
    other: OtherWeights = field(default_factory=OtherWeights)


@dataclass(frozen=True)
class StopValveConfig:
    min_h2h_pa: int = 2
    hitter_long_ba_min: float = 0.270
    hitter_recent_500_ba_min: float = 0.270
    hitter_last_25_ab_min: float = 0.150
    hitter_bb_rate_season_max: float = 0.120
    hitter_bb_rate_500_pa_max: float = 0.120
    hitter_pa_per_game_season_min: float = 4.200
    hitter_whiff_rate_season_max: float = 0.250
    hitter_whiff_rate_500_pa_max: float = 0.250
    hitter_k_rate_season_max: float = 0.220
    hitter_k_rate_500_pa_max: float = 0.220
    precipitation_probability_max: float = 40.0
    h2h_whiff_rate_max: float = 0.250
    h2h_k_rate_max: float = 0.200
    pitcher_hpi_200_min: float = 0.875
    pitcher_hpi_season_min: float = 0.875
    pitcher_hits_last_18_ip_min: int = 10
    pitcher_dominant_start_ip_min: float = 6.0
    pitcher_dominant_start_strikeouts_min: int = 8
    pitcher_dominant_start_hits_max: int = 3
    pitcher_dominant_start_walks_max: int = 2
    pitcher_stuff_plus_max: float = 95.0
    hitter_both_hands_min: float = 0.265
    hitter_both_hands_season_pa_min: int = 50
    hitter_matchup_hand_min: float = 0.270
    pitcher_split_opp_ba_min: float = 0.245
    strict_missing_stop_data: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    stop_valves: StopValveConfig = field(default_factory=StopValveConfig)
    min_starts_last_5: int = 3
    recent_usage_games: int = 5
    savant_years: int = 3
    hitter_play_log_seasons_back: int = 7
    pitcher_game_log_seasons_back: int = 6
    request_sleep_seconds: float = 0.03
    compute_bullpen: bool = True
    use_weather: bool = True
    use_fangraphs_fetch: bool = True
    stuff_plus_csv: Path = DEFAULT_STUFF_PLUS_CSV
    output_csv: Path = DEFAULT_OUTPUT_CSV
