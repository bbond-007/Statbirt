from __future__ import annotations

from collections.abc import Iterable

from .config import PipelineConfig, ScoreWeights, StopValveConfig
from .models import CandidateFeatures, ScoreComponent, ScoredCandidate, ValveResult
from .utils import clamp, parse_float, score_on_scale, weighted_average


def _add_component(components: list[ScoreComponent], name: str, weight: float, raw_value, subscore: float):
    points = (weight / 100.0) * subscore
    components.append(
        ScoreComponent(
            name=name,
            weight=weight,
            raw_value=raw_value,
            subscore=round(subscore, 2),
            points=round(points, 3),
        )
    )


def expected_pa_score(lineup_slot: float | None, expected_pa: float | None) -> float:
    if expected_pa is not None:
        return score_on_scale(expected_pa, 3.6, 4.8, higher_is_better=True)
    if lineup_slot is None:
        return 50.0
    return score_on_scale(lineup_slot, 1.0, 9.0, higher_is_better=False)


def direct_h2h_score(features: CandidateFeatures) -> float:
    return weighted_average(
        [
            (score_on_scale(features.h2h_hit_rate, 0.100, 0.420), 2.0),
            (score_on_scale(features.h2h_xba, 0.180, 0.380), 2.0),
            (score_on_scale(features.h2h_exit_velocity, 82.0, 96.0), 1.0),
            (score_on_scale(features.h2h_whiff_rate, 0.050, 0.350, higher_is_better=False), 1.25),
            (score_on_scale(features.h2h_k_rate, 0.050, 0.350, higher_is_better=False), 1.25),
        ],
        default=50.0,
    )


def inferred_pitch_type_score(features: CandidateFeatures) -> float:
    return weighted_average(
        [
            (score_on_scale(features.inferred_pitch_type_ba, 0.180, 0.360), 1.0),
            (score_on_scale(features.inferred_pitch_type_xba, 0.180, 0.360), 1.25),
        ],
        default=50.0,
    )


def score_features(features: CandidateFeatures, weights: ScoreWeights | None = None) -> tuple[float, tuple[ScoreComponent, ...]]:
    weights = weights or ScoreWeights()
    components: list[ScoreComponent] = []

    _add_component(
        components,
        "hitter.hipa_2500_pa",
        weights.hitter.hipa_2500_pa,
        features.hitter_hipa_2500_pa,
        score_on_scale(features.hitter_hipa_2500_pa, 0.175, 0.255),
    )
    _add_component(
        components,
        "hitter.pa_per_game_season",
        weights.hitter.pa_per_game_season,
        features.hitter_pa_per_game_season,
        score_on_scale(features.hitter_pa_per_game_season, 4.0, 4.9),
    )
    _add_component(
        components,
        "hitter.hipa_500_pa",
        weights.hitter.hipa_500_pa,
        features.hitter_hipa_500_pa,
        score_on_scale(features.hitter_hipa_500_pa, 0.170, 0.270),
    )
    _add_component(
        components,
        "hitter.hipa_75_ab",
        weights.hitter.hipa_75_ab,
        features.hitter_hipa_75_ab,
        score_on_scale(features.hitter_hipa_75_ab, 0.145, 0.295),
    )

    _add_component(
        components,
        "starting_pitcher.hpi_350",
        weights.starting_pitcher.hits_per_inning_350,
        features.pitcher_hpi_350,
        score_on_scale(features.pitcher_hpi_350, 0.750, 1.250),
    )
    _add_component(
        components,
        "starting_pitcher.hpi_season",
        weights.starting_pitcher.hits_per_inning_season,
        features.pitcher_hpi_season,
        score_on_scale(features.pitcher_hpi_season, 0.750, 1.250),
    )
    _add_component(
        components,
        "starting_pitcher.stuff_plus",
        weights.starting_pitcher.stuff_plus,
        features.pitcher_stuff_plus,
        score_on_scale(features.pitcher_stuff_plus, 80.0, 115.0, higher_is_better=False),
    )

    _add_component(
        components,
        "h2h.direct",
        weights.h2h.direct_matchup,
        features.h2h_pa,
        direct_h2h_score(features),
    )
    _add_component(
        components,
        "h2h.pitcher_lr_opp_ba",
        weights.h2h.pitcher_lr_split,
        features.pitcher_lr_opp_ba,
        score_on_scale(features.pitcher_lr_opp_ba, 0.220, 0.330),
    )
    _add_component(
        components,
        "h2h.inferred_pitch_type",
        weights.h2h.inferred_pitch_type,
        features.inferred_pitch_type_xba,
        inferred_pitch_type_score(features),
    )

    _add_component(
        components,
        "bullpen.hpi_season",
        weights.bullpen.hits_per_inning_season,
        features.bullpen_hpi,
        score_on_scale(features.bullpen_hpi, 0.750, 1.200),
    )

    _add_component(
        components,
        "other.road_game",
        weights.other.road_game,
        not features.is_home,
        100.0 if not features.is_home else 0.0,
    )
    _add_component(
        components,
        "other.division_matchup",
        weights.other.division_matchup,
        features.same_division,
        100.0 if features.same_division else 0.0,
    )
    _add_component(
        components,
        "other.sprint_speed",
        weights.other.sprint_speed,
        features.sprint_speed,
        score_on_scale(features.sprint_speed, 24.5, 30.0),
    )
    _add_component(
        components,
        "other.park_hit_factor",
        weights.other.park_hit_factor,
        features.park_hit_factor,
        score_on_scale(features.park_hit_factor, 90.0, 110.0),
    )
    _add_component(
        components,
        "other.lineup_opportunity",
        weights.other.lineup_opportunity,
        features.lineup_slot or features.expected_pa,
        expected_pa_score(features.lineup_slot, features.expected_pa),
    )

    score = round(sum(component.points for component in components), 2)
    return clamp(score, 0.0, 100.0), tuple(components)


def _missing_or_under(value, threshold: float) -> bool | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return parsed < threshold


def _missing_or_over(value, threshold: float) -> bool | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return parsed > threshold


def _add_missing_or_hard(
    outcome: bool | None,
    *,
    reason: str,
    missing_reason: str,
    hard: list[str],
    concerns: list[str],
    strict_missing: bool,
):
    if outcome is True:
        hard.append(reason)
    elif outcome is None:
        (hard if strict_missing else concerns).append(missing_reason)


def _split_ok(values: Iterable[float | None], threshold: float) -> bool | None:
    seen = False
    for value in values:
        parsed = parse_float(value)
        if parsed is None:
            continue
        seen = True
        if parsed >= threshold:
            return True
    return False if seen else None


def evaluate_stop_valves(features: CandidateFeatures, config: StopValveConfig | None = None) -> ValveResult:
    config = config or StopValveConfig()
    hard: list[str] = []
    concerns: list[str] = list(features.missing_data)

    if (features.h2h_pa or 0) < config.min_h2h_pa:
        hard.append(f"H2H PA below {config.min_h2h_pa}")
    if features.doubleheader:
        hard.append("Hitter is playing in a doubleheader")
    if features.opener_risk:
        hard.append("Opposing team may be using an opener")

    long_low = _missing_or_under(features.hitter_ba_2500_ab, config.hitter_long_ba_min)
    recent_low = _missing_or_under(features.hitter_ba_500_ab, config.hitter_recent_500_ba_min)
    if long_low is True and recent_low is True:
        hard.append("Hitter is below .270 over both long and last-500 AB windows")
    elif long_low is None or recent_low is None:
        concerns.append("Missing one hitter BA window for .270 long/500 AB stop valve")

    _add_missing_or_hard(
        _missing_or_under(features.hitter_ba_25_ab, config.hitter_last_25_ab_min),
        reason="Hitter BA under .150 over last 25 AB",
        missing_reason="Missing hitter last-25 AB BA",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_over(features.hitter_bb_rate_season, config.hitter_bb_rate_season_max),
        reason="Hitter season BB rate over 12%",
        missing_reason="Missing hitter season BB rate",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_over(features.hitter_bb_rate_500_pa, config.hitter_bb_rate_500_pa_max),
        reason="Hitter last-500 PA BB rate over 12%",
        missing_reason="Missing hitter last-500 PA BB rate",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_under(features.hitter_pa_per_game_season, config.hitter_pa_per_game_season_min),
        reason="Hitter season PA/G under 4.2",
        missing_reason="Missing hitter season PA/G",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_over(features.hitter_whiff_rate_season, config.hitter_whiff_rate_season_max),
        reason="Hitter season whiff rate over 25%",
        missing_reason="Missing hitter season whiff rate",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_over(features.hitter_whiff_rate_500_pa, config.hitter_whiff_rate_500_pa_max),
        reason="Hitter last-500 PA whiff rate over 25%",
        missing_reason="Missing hitter last-500 PA whiff rate",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_over(features.hitter_k_rate_season, config.hitter_k_rate_season_max),
        reason="Hitter season K rate over 22%",
        missing_reason="Missing hitter season K rate",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_over(features.hitter_k_rate_500_pa, config.hitter_k_rate_500_pa_max),
        reason="Hitter last-500 PA K rate over 22%",
        missing_reason="Missing hitter last-500 PA K rate",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_over(features.precipitation_probability, config.precipitation_probability_max),
        reason="Precipitation probability over 40%",
        missing_reason="Missing weather precipitation probability",
        hard=hard,
        concerns=concerns,
        strict_missing=False,
    )
    _add_missing_or_hard(
        _missing_or_over(features.h2h_whiff_rate, config.h2h_whiff_rate_max),
        reason="H2H whiff rate over 25%",
        missing_reason="Missing H2H whiff rate",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_over(features.h2h_k_rate, config.h2h_k_rate_max),
        reason="H2H K rate over 20%",
        missing_reason="Missing H2H K rate",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_under(features.pitcher_hpi_200, config.pitcher_hpi_200_min),
        reason="Starter H/IP under .875 over last 200 innings",
        missing_reason="Missing starter last-200 IP H/IP",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    _add_missing_or_hard(
        _missing_or_under(features.pitcher_hpi_season, config.pitcher_hpi_season_min),
        reason="Starter season H/IP under .875",
        missing_reason="Missing starter season H/IP",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )
    hits_last_18 = parse_float(features.pitcher_hits_last_18_ip)
    if hits_last_18 is None:
        (hard if config.strict_missing_stop_data else concerns).append("Missing starter hits over last 18 IP")
    elif hits_last_18 < config.pitcher_hits_last_18_ip_min:
        hard.append("Starter allowed fewer than 10 hits over last 18 IP")

    _add_missing_or_hard(
        _missing_or_over(features.pitcher_stuff_plus, config.pitcher_stuff_plus_max),
        reason="Starter Stuff+ over 95",
        missing_reason="Missing starter Stuff+",
        hard=hard,
        concerns=concerns,
        strict_missing=config.strict_missing_stop_data,
    )

    left_ok = _split_ok(
        [features.hitter_split_ba_500_vs_lhp, features.hitter_split_ba_1500_vs_lhp],
        config.hitter_both_hands_min,
    )
    right_ok = _split_ok(
        [features.hitter_split_ba_500_vs_rhp, features.hitter_split_ba_1500_vs_rhp],
        config.hitter_both_hands_min,
    )
    if left_ok is False or right_ok is False:
        hard.append("Hitter is not at least .265 against both pitcher hands in 500/1500 AB windows")
    elif left_ok is None or right_ok is None:
        (hard if config.strict_missing_stop_data else concerns).append(
            "Missing hitter both-hands split stop-valve data"
        )

    matchup_hand_ok = _split_ok(
        [features.hitter_matchup_hand_ba_500, features.hitter_matchup_hand_ba_1500],
        config.hitter_matchup_hand_min,
    )
    if matchup_hand_ok is False:
        hard.append("Hitter is not at least .270 against today's pitcher hand")
    elif matchup_hand_ok is None:
        (hard if config.strict_missing_stop_data else concerns).append(
            "Missing hitter matchup-hand split stop-valve data"
        )

    split_50_low = _missing_or_under(features.pitcher_lr_opp_ba_50, config.pitcher_split_opp_ba_min)
    split_200_low = _missing_or_under(features.pitcher_lr_opp_ba_200, config.pitcher_split_opp_ba_min)
    if split_50_low is True or split_200_low is True:
        hard.append("Pitcher is not allowing over .245 BA to hitter hand in both 50/200 IP windows")
    elif split_50_low is None or split_200_low is None:
        (hard if config.strict_missing_stop_data else concerns).append(
            "Missing pitcher hand-split 50/200 IP stop-valve data"
        )

    deduped_hard = tuple(dict.fromkeys(hard))
    deduped_concerns = tuple(reason for reason in dict.fromkeys(concerns) if reason not in deduped_hard)
    return ValveResult(pickable=not deduped_hard, hard_pass_reasons=deduped_hard, concerns=deduped_concerns)


def score_candidate(features: CandidateFeatures, config: PipelineConfig | None = None) -> ScoredCandidate:
    config = config or PipelineConfig()
    score, components = score_features(features, config.weights)
    valve_result = evaluate_stop_valves(features, config.stop_valves)
    return ScoredCandidate(
        features=features,
        score=score,
        components=components,
        valve_result=valve_result,
    )
