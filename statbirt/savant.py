from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import io
import json
import math
import re
import time
from typing import Iterable

import pandas as pd
import requests

from .cache import cache_path, load_pickle, save_pickle
from .utils import (
    HIT_EVENTS,
    NON_AB_EVENTS,
    STRIKEOUT_EVENTS,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
    clamp,
    parse_float,
    parse_int,
    safe_divide,
    stable_hash,
    weighted_average,
)

STATCAST_SEARCH_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
PARK_FACTORS_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
HEADERS = {"User-Agent": "Mozilla/5.0"}
STATCAST_CAREER_START_DATE = date(2015, 3, 1)
REQUEST_TIMEOUT_SECONDS = 90
REQUEST_SLEEP_SECONDS = 0.08
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_REQUEST_ATTEMPTS = 4
MATCHUP_CACHE_SCHEMA = 4


@dataclass(frozen=True)
class SeasonWindow:
    season: int
    start_date: date
    end_date: date
    weight: float = 1.0


@dataclass(frozen=True)
class H2HFeatures:
    pa: int = 0
    hits: int = 0
    hit_rate: float | None = None
    whiff_rate: float | None = None
    k_rate: float | None = None
    exit_velocity: float | None = None
    xba: float | None = None


@dataclass(frozen=True)
class InferredPitchTypeFeatures:
    ba: float | None = None
    xba: float | None = None
    contact_rate: float | None = None
    shape_distance: float | None = None
    coverage: float = 0.0


def _base_params(player_type: str, start_dt: str, end_dt: str) -> list[tuple[str, str]]:
    return [
        ("all", "true"),
        ("hfPT", ""),
        ("hfAB", ""),
        ("hfBBT", ""),
        ("hfPR", ""),
        ("hfZ", ""),
        ("stadium", ""),
        ("hfBBL", ""),
        ("hfNewZones", ""),
        ("hfGT", "R|PO|S|"),
        ("hfSea", ""),
        ("hfSit", ""),
        ("player_type", player_type),
        ("hfOuts", ""),
        ("opponent", ""),
        ("pitcher_throws", ""),
        ("batter_stands", ""),
        ("hfSA", ""),
        ("game_date_gt", start_dt),
        ("game_date_lt", end_dt),
        ("team", ""),
        ("position", ""),
        ("hfRO", ""),
        ("home_road", ""),
        ("hfFlag", ""),
        ("metric_1", ""),
        ("hfInn", ""),
        ("min_pitches", "0"),
        ("min_results", "0"),
        ("group_by", "name"),
        ("sort_col", "pitches"),
        ("player_event_sort", "h_launch_speed"),
        ("sort_order", "desc"),
        ("min_abs", "0"),
        ("type", "details"),
    ]


def _read_statcast_csv(text: str) -> pd.DataFrame:
    if not text:
        return pd.DataFrame()
    if text.lstrip().startswith("<"):
        raise ValueError("Baseball Savant returned HTML instead of CSV.")
    return pd.read_csv(io.StringIO(text), low_memory=False)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        return bool(response is not None and response.status_code in RETRYABLE_STATUS_CODES)
    return isinstance(exc, (ValueError, pd.errors.EmptyDataError))


def _fetch_statcast_batch(
    *,
    session: requests.Session,
    player_type: str,
    player_key: str,
    batch: list[int],
    window: SeasonWindow,
) -> pd.DataFrame:
    params = _base_params(player_type, window.start_date.isoformat(), window.end_date.isoformat())
    params.extend((player_key, str(player_id)) for player_id in batch)
    last_exc: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = session.get(STATCAST_SEARCH_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            text = response.content.decode("utf-8-sig").strip()
            return _read_statcast_csv(text)
        except (requests.exceptions.RequestException, ValueError, pd.errors.EmptyDataError) as exc:
            last_exc = exc
            if attempt >= MAX_REQUEST_ATTEMPTS or not _is_retryable(exc):
                break
            time.sleep(1.5 * (2 ** (attempt - 1)))

    if last_exc is not None and len(batch) > 1:
        midpoint = max(1, len(batch) // 2)
        left = _fetch_statcast_batch(
            session=session,
            player_type=player_type,
            player_key=player_key,
            batch=batch[:midpoint],
            window=window,
        )
        right = _fetch_statcast_batch(
            session=session,
            player_type=player_type,
            player_key=player_key,
            batch=batch[midpoint:],
            window=window,
        )
        if left.empty:
            return right
        if right.empty:
            return left
        return pd.concat([left, right], ignore_index=True)
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def chunked(values: Iterable[int], size: int):
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_statcast_details(
    *,
    player_type: str,
    player_ids: Iterable[int],
    windows: Iterable[SeasonWindow],
    batch_size: int,
) -> pd.DataFrame:
    ids = sorted({int(player_id) for player_id in player_ids if player_id is not None})
    if not ids:
        return pd.DataFrame()
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    player_key = "batters_lookup[]" if player_type == "batter" else "pitchers_lookup[]"
    for window in windows:
        if window.end_date < window.start_date:
            continue
        for batch in chunked(ids, batch_size):
            frame = _fetch_statcast_batch(
                session=session,
                player_type=player_type,
                player_key=player_key,
                batch=batch,
                window=window,
            )
            if frame.empty:
                continue
            frame = frame.copy()
            frame["season_window"] = window.season
            frame["season_weight"] = window.weight
            frames.append(frame)
            time.sleep(REQUEST_SLEEP_SECONDS)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_statcast_matchup_batch(
    *,
    session: requests.Session,
    batter_batch: list[int],
    pitcher_batch: list[int],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    params = _base_params("pitcher", start_date.isoformat(), end_date.isoformat())
    params.extend(("pitchers_lookup[]", str(player_id)) for player_id in pitcher_batch)
    params.extend(("batters_lookup[]", str(player_id)) for player_id in batter_batch)
    last_exc: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = session.get(STATCAST_SEARCH_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            text = response.content.decode("utf-8-sig").strip()
            return _read_statcast_csv(text)
        except (requests.exceptions.RequestException, ValueError, pd.errors.EmptyDataError) as exc:
            last_exc = exc
            if attempt >= MAX_REQUEST_ATTEMPTS or not _is_retryable(exc):
                break
            time.sleep(1.5 * (2 ** (attempt - 1)))

    if last_exc is not None and (len(batter_batch) > 1 or len(pitcher_batch) > 1):
        if len(pitcher_batch) >= len(batter_batch) and len(pitcher_batch) > 1:
            midpoint = max(1, len(pitcher_batch) // 2)
            left = _fetch_statcast_matchup_batch(
                session=session,
                batter_batch=batter_batch,
                pitcher_batch=pitcher_batch[:midpoint],
                start_date=start_date,
                end_date=end_date,
            )
            right = _fetch_statcast_matchup_batch(
                session=session,
                batter_batch=batter_batch,
                pitcher_batch=pitcher_batch[midpoint:],
                start_date=start_date,
                end_date=end_date,
            )
        else:
            midpoint = max(1, len(batter_batch) // 2)
            left = _fetch_statcast_matchup_batch(
                session=session,
                batter_batch=batter_batch[:midpoint],
                pitcher_batch=pitcher_batch,
                start_date=start_date,
                end_date=end_date,
            )
            right = _fetch_statcast_matchup_batch(
                session=session,
                batter_batch=batter_batch[midpoint:],
                pitcher_batch=pitcher_batch,
                start_date=start_date,
                end_date=end_date,
            )
        if left.empty:
            return right
        if right.empty:
            return left
        return pd.concat([left, right], ignore_index=True)
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def fetch_statcast_matchup_details(
    *,
    batter_ids: Iterable[int],
    pitcher_ids: Iterable[int],
    start_date: date,
    end_date: date,
    batter_batch_size: int,
    pitcher_batch_size: int,
) -> pd.DataFrame:
    batters = sorted({int(player_id) for player_id in batter_ids if player_id is not None})
    pitchers = sorted({int(player_id) for player_id in pitcher_ids if player_id is not None})
    if not batters or not pitchers or end_date < start_date:
        return pd.DataFrame()
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    for pitcher_batch in chunked(pitchers, pitcher_batch_size):
        for batter_batch in chunked(batters, batter_batch_size):
            frame = _fetch_statcast_matchup_batch(
                session=session,
                batter_batch=batter_batch,
                pitcher_batch=pitcher_batch,
                start_date=start_date,
                end_date=end_date,
            )
            if frame.empty:
                continue
            frames.append(frame)
            time.sleep(REQUEST_SLEEP_SECONDS)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def prepare_pitch_data(frame: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "batter",
        "pitcher",
        "pitch_type",
        "stand",
        "p_throws",
        "events",
        "description",
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "estimated_ba_using_speedangle",
        "bat_speed",
        "swing_length",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "season_window",
        "season_weight",
        "is_hit",
        "is_ab",
        "is_k",
    ]
    if frame.empty:
        return pd.DataFrame(columns=expected_columns)
    cleaned = frame.copy()

    def series_or_default(column: str, default_value):
        if column in cleaned.columns:
            return cleaned[column]
        return pd.Series([default_value] * len(cleaned), index=cleaned.index)

    for optional_numeric in (
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "estimated_ba_using_speedangle",
        "bat_speed",
        "swing_length",
        "release_speed",
        "pfx_x",
        "pfx_z",
    ):
        if optional_numeric not in cleaned.columns:
            cleaned[optional_numeric] = None

    _to_numeric(
        cleaned,
        [
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "batter",
            "pitcher",
            "launch_speed",
            "launch_angle",
            "launch_speed_angle",
            "estimated_ba_using_speedangle",
            "bat_speed",
            "swing_length",
            "release_speed",
            "pfx_x",
            "pfx_z",
            "season_window",
            "season_weight",
        ],
    )
    cleaned["game_date"] = pd.to_datetime(series_or_default("game_date", None), errors="coerce").dt.date
    cleaned["pitch_type"] = series_or_default("pitch_type", "UNK").fillna("UNK").astype(str).str.upper()
    cleaned["stand"] = series_or_default("stand", "?").fillna("?").astype(str).str.upper()
    cleaned["p_throws"] = series_or_default("p_throws", "?").fillna("?").astype(str).str.upper()
    cleaned["events"] = series_or_default("events", "").fillna("").astype(str).str.lower()
    cleaned["description"] = series_or_default("description", "").fillna("").astype(str).str.lower()
    cleaned["season_weight"] = pd.to_numeric(series_or_default("season_weight", 1.0), errors="coerce").fillna(1.0)
    cleaned["season_window"] = pd.to_numeric(series_or_default("season_window", None), errors="coerce")
    cleaned["is_hit"] = cleaned["events"].isin(HIT_EVENTS)
    cleaned["is_ab"] = (cleaned["events"] != "") & ~cleaned["events"].isin(NON_AB_EVENTS)
    cleaned["is_k"] = cleaned["events"].isin(STRIKEOUT_EVENTS)
    return cleaned[expected_columns].copy()


def final_pa_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"game_pk", "at_bat_number", "pitch_number", "events"}
    if frame.empty or not required.issubset(set(frame.columns)):
        return pd.DataFrame()
    valid = frame[
        frame["game_pk"].notna()
        & frame["at_bat_number"].notna()
        & frame["pitch_number"].notna()
        & (frame["events"].fillna("") != "")
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    valid = valid.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    return valid.groupby(["game_pk", "at_bat_number"], as_index=False).tail(1).copy()


def _xba_mean(frame: pd.DataFrame) -> float | None:
    values = pd.to_numeric(frame.get("estimated_ba_using_speedangle"), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _ev_mean(frame: pd.DataFrame) -> float | None:
    values = pd.to_numeric(frame.get("launch_speed"), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _numeric_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame.get(column), errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


class StatcastFeatureStore:
    def __init__(self, batter_df: pd.DataFrame, pitcher_df: pd.DataFrame, h2h_df: pd.DataFrame | None = None):
        self.batter_df = prepare_pitch_data(batter_df)
        self.pitcher_df = prepare_pitch_data(pitcher_df)
        self.h2h_df = prepare_pitch_data(h2h_df) if h2h_df is not None and not h2h_df.empty else self.pitcher_df
        self.batter_pa = final_pa_rows(self.batter_df)
        self.pitcher_pa = final_pa_rows(self.pitcher_df)
        self.h2h_pa = final_pa_rows(self.h2h_df)

    def h2h(self, batter_id: int, pitcher_id: int | None) -> H2HFeatures:
        if pitcher_id is None or self.h2h_df.empty:
            return H2HFeatures()
        pitches = self.h2h_df[
            (self.h2h_df["batter"] == batter_id)
            & (self.h2h_df["pitcher"] == pitcher_id)
        ].copy()
        pa = self.h2h_pa[
            (self.h2h_pa["batter"] == batter_id)
            & (self.h2h_pa["pitcher"] == pitcher_id)
        ].copy()
        if pa.empty:
            return H2HFeatures()
        swings = pitches["description"].isin(SWING_DESCRIPTIONS).sum() if not pitches.empty else 0
        whiffs = pitches["description"].isin(WHIFF_DESCRIPTIONS).sum() if not pitches.empty else 0
        hits = int(pa["is_hit"].sum())
        return H2HFeatures(
            pa=int(len(pa)),
            hits=hits,
            hit_rate=safe_divide(hits, len(pa)),
            whiff_rate=safe_divide(whiffs, swings) if swings else None,
            k_rate=safe_divide(int(pa["is_k"].sum()), len(pa)),
            exit_velocity=_ev_mean(pa),
            xba=_xba_mean(pa),
        )

    def hitter_split_windows(self, batter_id: int, *, target_date: date) -> dict[str, float | int | None]:
        pa = self.batter_pa[self.batter_pa["batter"] == batter_id].copy()
        if pa.empty:
            return {
                "ba_season_vs_lhp": None,
                "ba_season_vs_rhp": None,
                "pa_season_vs_lhp": None,
                "pa_season_vs_rhp": None,
                "ba_500_vs_lhp": None,
                "ba_500_vs_rhp": None,
                "ba_1500_vs_lhp": None,
                "ba_1500_vs_rhp": None,
            }
        pa = pa[pa["game_date"] < target_date].sort_values(
            ["game_date", "game_pk", "at_bat_number"], ascending=False
        )

        def split_ba(hand: str, ab_window: int):
            subset = pa[(pa["p_throws"] == hand) & (pa["is_ab"])]
            if subset.empty:
                return None
            used = subset.head(ab_window)
            at_bats = len(used)
            return safe_divide(int(used["is_hit"].sum()), at_bats)

        season_years = pa["game_date"].map(lambda value: getattr(value, "year", None))

        def season_split(hand: str):
            subset = pa[(pa["p_throws"] == hand) & (season_years == target_date.year)]
            plate_appearances = len(subset)
            at_bats = int(subset["is_ab"].sum()) if plate_appearances else 0
            hits = int(subset["is_hit"].sum()) if plate_appearances else 0
            return safe_divide(hits, at_bats), plate_appearances or None

        season_ba_lhp, season_pa_lhp = season_split("L")
        season_ba_rhp, season_pa_rhp = season_split("R")

        return {
            "ba_season_vs_lhp": season_ba_lhp,
            "ba_season_vs_rhp": season_ba_rhp,
            "pa_season_vs_lhp": season_pa_lhp,
            "pa_season_vs_rhp": season_pa_rhp,
            "ba_500_vs_lhp": split_ba("L", 500),
            "ba_500_vs_rhp": split_ba("R", 500),
            "ba_1500_vs_lhp": split_ba("L", 1500),
            "ba_1500_vs_rhp": split_ba("R", 1500),
        }

    def hitter_plate_discipline(
        self,
        batter_id: int,
        *,
        target_date: date,
        current_season: int,
        pa_window: int = 500,
    ) -> dict[str, float | None]:
        pa = self.batter_pa[self.batter_pa["batter"] == batter_id].copy()
        pitches = self.batter_df[self.batter_df["batter"] == batter_id].copy()
        if pa.empty:
            return {
                "bb_rate_season": None,
                "bb_rate_500_pa": None,
                "whiff_rate_season": None,
                "whiff_rate_500_pa": None,
                "k_rate_season": None,
                "k_rate_500_pa": None,
            }
        pa = pa[pa["game_date"] < target_date].sort_values(
            ["game_date", "game_pk", "at_bat_number"], ascending=False
        )
        pitches = pitches[pitches["game_date"] < target_date].copy()
        season_pa = pa[pa["season_window"] == current_season]
        season_pitches = pitches[pitches["season_window"] == current_season]
        recent_pa = pa.head(pa_window)
        recent_keys = {
            (int(row.game_pk), int(row.at_bat_number))
            for row in recent_pa.itertuples()
            if not pd.isna(row.game_pk) and not pd.isna(row.at_bat_number)
        }
        if recent_keys and not pitches.empty:
            recent_pitches = pitches[
                pitches.apply(
                    lambda row: (int(row["game_pk"]), int(row["at_bat_number"])) in recent_keys
                    if not pd.isna(row["game_pk"]) and not pd.isna(row["at_bat_number"])
                    else False,
                    axis=1,
                )
            ]
        else:
            recent_pitches = pitches.iloc[0:0]

        def rates(pa_subset: pd.DataFrame, pitch_subset: pd.DataFrame):
            if pa_subset.empty:
                return None, None, None
            pa_count = len(pa_subset)
            walks = pa_subset["events"].isin({"walk", "intent_walk"}).sum()
            strikeouts = pa_subset["is_k"].sum()
            swings = pitch_subset["description"].isin(SWING_DESCRIPTIONS).sum() if not pitch_subset.empty else 0
            whiffs = pitch_subset["description"].isin(WHIFF_DESCRIPTIONS).sum() if not pitch_subset.empty else 0
            return (
                safe_divide(int(walks), pa_count),
                safe_divide(int(whiffs), int(swings)) if swings else None,
                safe_divide(int(strikeouts), pa_count),
            )

        season_bb, season_whiff, season_k = rates(season_pa, season_pitches)
        recent_bb, recent_whiff, recent_k = rates(recent_pa, recent_pitches)
        return {
            "bb_rate_season": season_bb,
            "bb_rate_500_pa": recent_bb,
            "whiff_rate_season": season_whiff,
            "whiff_rate_500_pa": recent_whiff,
            "k_rate_season": season_k,
            "k_rate_500_pa": recent_k,
        }

    @staticmethod
    def _contact_window(
        frame: pd.DataFrame,
        *,
        bbe_window: int = 50,
        hardest_ev50: bool = True,
    ) -> dict[str, float | int | None]:
        if frame.empty:
            return {
                "contact_xba": None,
                "hard_hit_rate": None,
                "sweet_spot_rate": None,
                "ev50": None,
                "bbe": 0,
            }
        batted = frame[pd.to_numeric(frame["launch_speed"], errors="coerce").notna()].copy()
        batted = batted.sort_values(
            ["game_date", "game_pk", "at_bat_number"],
            ascending=False,
        ).head(bbe_window)
        if batted.empty:
            return {
                "contact_xba": None,
                "hard_hit_rate": None,
                "sweet_spot_rate": None,
                "ev50": None,
                "bbe": 0,
            }
        exit_velocity = pd.to_numeric(batted["launch_speed"], errors="coerce").dropna()
        launch_angle = pd.to_numeric(batted["launch_angle"], errors="coerce").dropna()
        hardest_count = max(1, int(math.ceil(len(exit_velocity) / 2.0)))
        return {
            "contact_xba": _xba_mean(batted),
            "hard_hit_rate": float((exit_velocity >= 95.0).mean()) if not exit_velocity.empty else None,
            "sweet_spot_rate": (
                float(launch_angle.between(8.0, 32.0, inclusive="both").mean())
                if not launch_angle.empty
                else None
            ),
            "ev50": (
                float(
                    (
                        exit_velocity.nlargest(hardest_count)
                        if hardest_ev50
                        else exit_velocity.nsmallest(hardest_count)
                    ).mean()
                )
                if not exit_velocity.empty
                else None
            ),
            "bbe": int(len(batted)),
        }

    @staticmethod
    def _xba_pa_window(frame: pd.DataFrame, *, pa_window: int = 100) -> dict[str, float | int | None]:
        if frame.empty:
            return {"xba": None, "xba_denominator": 0}
        recent = frame.sort_values(
            ["game_date", "game_pk", "at_bat_number"],
            ascending=False,
        ).head(pa_window)
        tracked = pd.to_numeric(recent["estimated_ba_using_speedangle"], errors="coerce").dropna()
        strikeouts = int(recent["is_k"].sum())
        denominator = int(len(tracked)) + strikeouts
        return {
            "xba": float(tracked.sum() / denominator) if denominator else None,
            "xba_denominator": denominator,
        }

    @staticmethod
    def _swing_window(frame: pd.DataFrame, *, swing_window: int = 50) -> dict[str, float | int | None]:
        if frame.empty:
            return {"bat_speed": None, "swing_length": None, "swings": 0}
        swings = frame[
            pd.to_numeric(frame["bat_speed"], errors="coerce").notna()
            | pd.to_numeric(frame["swing_length"], errors="coerce").notna()
        ].copy()
        swings = swings.sort_values(
            ["game_date", "game_pk", "at_bat_number", "pitch_number"],
            ascending=False,
        ).head(swing_window)
        if swings.empty:
            return {"bat_speed": None, "swing_length": None, "swings": 0}
        bat_speed = pd.to_numeric(swings["bat_speed"], errors="coerce").dropna()
        swing_length = pd.to_numeric(swings["swing_length"], errors="coerce").dropna()
        return {
            "bat_speed": float(bat_speed.mean()) if not bat_speed.empty else None,
            "swing_length": float(swing_length.mean()) if not swing_length.empty else None,
            "swings": int(len(swings)),
        }

    def hitter_contact_quality(
        self,
        batter_id: int,
        *,
        target_date: date,
        bbe_window: int = 50,
        swing_window: int = 50,
    ) -> dict[str, float | int | None]:
        pa = self.batter_pa[
            (self.batter_pa["batter"] == batter_id) & (self.batter_pa["game_date"] < target_date)
        ].copy()
        pitches = self.batter_df[
            (self.batter_df["batter"] == batter_id) & (self.batter_df["game_date"] < target_date)
        ].copy()
        return {
            **self._contact_window(pa, bbe_window=bbe_window, hardest_ev50=True),
            **self._xba_pa_window(pa, pa_window=100),
            **self._swing_window(pitches, swing_window=swing_window),
        }

    def pitcher_contact_quality_allowed(
        self,
        pitcher_id: int | None,
        *,
        target_date: date,
        bbe_window: int = 50,
    ) -> dict[str, float | int | None]:
        if pitcher_id is None:
            return {
                **self._contact_window(self.pitcher_pa.iloc[0:0], bbe_window=bbe_window, hardest_ev50=False),
                **self._xba_pa_window(self.pitcher_pa.iloc[0:0], pa_window=100),
            }
        pa = self.pitcher_pa[
            (self.pitcher_pa["pitcher"] == pitcher_id) & (self.pitcher_pa["game_date"] < target_date)
        ].copy()
        return {
            **self._contact_window(pa, bbe_window=bbe_window, hardest_ev50=False),
            **self._xba_pa_window(pa, pa_window=100),
        }

    def pitcher_split_opp_ba(
        self,
        pitcher_id: int | None,
        stand: str,
        *,
        cutoff_date: date | None = None,
    ) -> float | None:
        if pitcher_id is None or self.pitcher_pa.empty:
            return None
        subset = self.pitcher_pa[
            (self.pitcher_pa["pitcher"] == pitcher_id)
            & (self.pitcher_pa["stand"] == stand)
            & (self.pitcher_pa["is_ab"])
        ].copy()
        if cutoff_date is not None:
            subset = subset[subset["game_date"] >= cutoff_date]
        if subset.empty:
            return None
        return safe_divide(int(subset["is_hit"].sum()), len(subset))

    def _pitch_arsenal(
        self,
        pitcher_id: int | None,
        stand: str,
        *,
        target_date: date,
        pitch_window: int = 500,
    ) -> dict[str, dict[str, float | None]]:
        if pitcher_id is None or self.pitcher_df.empty:
            return {}
        subset = self.pitcher_df[
            (self.pitcher_df["pitcher"] == pitcher_id)
            & (self.pitcher_df["stand"] == stand)
            & (self.pitcher_df["pitch_type"] != "UNK")
            & (self.pitcher_df["game_date"] < target_date)
        ]
        if subset.empty:
            subset = self.pitcher_df[
                (self.pitcher_df["pitcher"] == pitcher_id)
                & (self.pitcher_df["pitch_type"] != "UNK")
                & (self.pitcher_df["game_date"] < target_date)
            ]
        if subset.empty:
            return {}
        subset = subset.sort_values(
            ["game_date", "game_pk", "at_bat_number", "pitch_number"],
            ascending=False,
        ).head(pitch_window)
        counts = subset.groupby("pitch_type").size()
        total = float(counts.sum())
        output = {}
        for pitch_type, count in counts.items():
            pitch_rows = subset[subset["pitch_type"] == pitch_type]
            output[pitch_type] = {
                "usage": float(count / total),
                "release_speed": _numeric_mean(pitch_rows, "release_speed"),
                "pfx_x": _numeric_mean(pitch_rows, "pfx_x"),
                "pfx_z": _numeric_mean(pitch_rows, "pfx_z"),
            }
        return output

    def _hitter_pitch_type_rates(
        self,
        batter_id: int,
        pitcher_hand: str,
        pitch_type: str,
        *,
        target_date: date,
    ) -> dict[str, float | int | None]:
        pa = self.batter_pa[
            (self.batter_pa["batter"] == batter_id)
            & (self.batter_pa["p_throws"] == pitcher_hand)
            & (self.batter_pa["pitch_type"] == pitch_type)
            & (self.batter_pa["game_date"] < target_date)
        ].copy()
        if pa.empty:
            return {"ba": None, "xba": None, "pa": 0, "contact_rate": None, "swings": 0}
        ab = pa[pa["is_ab"]]
        ba = safe_divide(int(ab["is_hit"].sum()), len(ab)) if not ab.empty else None
        pitches = self.batter_df[
            (self.batter_df["batter"] == batter_id)
            & (self.batter_df["p_throws"] == pitcher_hand)
            & (self.batter_df["pitch_type"] == pitch_type)
            & (self.batter_df["game_date"] < target_date)
        ].copy()
        swings = int(pitches["description"].isin(SWING_DESCRIPTIONS).sum())
        whiffs = int(pitches["description"].isin(WHIFF_DESCRIPTIONS).sum())
        return {
            "ba": ba,
            "xba": _xba_mean(pa),
            "pa": int(len(pa)),
            "contact_rate": safe_divide(swings - whiffs, swings) if swings else None,
            "swings": swings,
            "release_speed": _numeric_mean(pitches, "release_speed"),
            "pfx_x": _numeric_mean(pitches, "pfx_x"),
            "pfx_z": _numeric_mean(pitches, "pfx_z"),
        }

    def inferred_pitch_type(
        self,
        *,
        batter_id: int,
        pitcher_id: int | None,
        pitcher_hand: str,
        stand: str,
        target_date: date,
        min_pitch_usage: float = 0.05,
    ) -> InferredPitchTypeFeatures:
        arsenal = self._pitch_arsenal(pitcher_id, stand, target_date=target_date)
        if not arsenal:
            return InferredPitchTypeFeatures()
        hand_pa = self.batter_pa[
            (self.batter_pa["batter"] == batter_id)
            & (self.batter_pa["p_throws"] == pitcher_hand)
            & (self.batter_pa["game_date"] < target_date)
        ]
        hand_pitches = self.batter_df[
            (self.batter_df["batter"] == batter_id)
            & (self.batter_df["p_throws"] == pitcher_hand)
            & (self.batter_df["game_date"] < target_date)
        ]
        hand_ab = hand_pa[hand_pa["is_ab"]]
        prior_ba = safe_divide(int(hand_ab["is_hit"].sum()), len(hand_ab)) if not hand_ab.empty else 0.250
        prior_xba_value = _xba_mean(hand_pa)
        prior_xba = 0.250 if prior_xba_value is None else prior_xba_value
        prior_swings = int(hand_pitches["description"].isin(SWING_DESCRIPTIONS).sum())
        prior_whiffs = int(hand_pitches["description"].isin(WHIFF_DESCRIPTIONS).sum())
        prior_contact = safe_divide(prior_swings - prior_whiffs, prior_swings) if prior_swings else 0.75
        pairs_ba = []
        pairs_xba = []
        pairs_contact = []
        pairs_shape = []
        coverage = 0.0
        for pitch_type, arsenal_row in arsenal.items():
            usage = float(arsenal_row.get("usage") or 0.0)
            if usage < min_pitch_usage:
                continue
            rates = self._hitter_pitch_type_rates(
                batter_id,
                pitcher_hand,
                pitch_type,
                target_date=target_date,
            )
            pa_sample = int(rates.get("pa") or 0)
            if pa_sample <= 0:
                continue
            ba = rates.get("ba")
            xba = rates.get("xba")
            contact = rates.get("contact_rate")
            swings = int(rates.get("swings") or 0)
            shrunk_ba = ((float(ba) * pa_sample + prior_ba * 50.0) / (pa_sample + 50.0)) if ba is not None else prior_ba
            shrunk_xba = (
                (float(xba) * pa_sample + prior_xba * 50.0) / (pa_sample + 50.0)
                if xba is not None
                else prior_xba
            )
            shrunk_contact = (
                (float(contact) * swings + prior_contact * 100.0) / (swings + 100.0)
                if contact is not None
                else prior_contact
            )
            pairs_ba.append((shrunk_ba, usage))
            pairs_xba.append((shrunk_xba, usage))
            pairs_contact.append((shrunk_contact, usage))
            shape_values = []
            for field, scale in (("release_speed", 3.0), ("pfx_x", 0.25), ("pfx_z", 0.25)):
                thrown = parse_float(arsenal_row.get(field))
                seen = parse_float(rates.get(field))
                if thrown is not None and seen is not None:
                    shape_values.append(((thrown - seen) / scale) ** 2)
            if shape_values:
                pairs_shape.append((math.sqrt(sum(shape_values)), usage))
            coverage += usage
        return InferredPitchTypeFeatures(
            ba=weighted_average(pairs_ba, default=None),
            xba=weighted_average(pairs_xba, default=None),
            contact_rate=weighted_average(pairs_contact, default=None),
            shape_distance=weighted_average(pairs_shape, default=None),
            coverage=clamp(coverage, 0.0, 1.0),
        )


def load_or_build_statcast_store(
    *,
    batter_ids: Iterable[int],
    pitcher_ids: Iterable[int],
    windows: Iterable[SeasonWindow],
    batter_batch_size: int = 24,
    pitcher_batch_size: int = 12,
    h2h_start_date: date | None = None,
    h2h_end_date: date | None = None,
    h2h_batter_batch_size: int = 24,
    h2h_pitcher_batch_size: int = 12,
) -> tuple[StatcastFeatureStore, bool, str]:
    normalized_windows = [
        {
            "season": window.season,
            "start_date": window.start_date.isoformat(),
            "end_date": window.end_date.isoformat(),
            "weight": window.weight,
        }
        for window in windows
        if window.end_date >= window.start_date
    ]
    cache_key = stable_hash(
        {
            "schema": MATCHUP_CACHE_SCHEMA,
            "batter_ids": sorted({int(player_id) for player_id in batter_ids if player_id is not None}),
            "pitcher_ids": sorted({int(player_id) for player_id in pitcher_ids if player_id is not None}),
            "windows": normalized_windows,
            "batter_batch_size": batter_batch_size,
            "pitcher_batch_size": pitcher_batch_size,
            "h2h_start_date": h2h_start_date.isoformat() if h2h_start_date else None,
            "h2h_end_date": h2h_end_date.isoformat() if h2h_end_date else None,
            "h2h_batter_batch_size": h2h_batter_batch_size,
            "h2h_pitcher_batch_size": h2h_pitcher_batch_size,
        }
    )
    cache_file = cache_path("statcast_store", f"{cache_key}.pkl")
    cached = load_pickle(cache_file)
    if isinstance(cached, dict) and cached.get("schema") == MATCHUP_CACHE_SCHEMA:
        store = cached.get("store")
        if isinstance(store, StatcastFeatureStore):
            return store, True, str(cache_file)

    batter_df = fetch_statcast_details(
        player_type="batter",
        player_ids=batter_ids,
        windows=windows,
        batch_size=batter_batch_size,
    )
    pitcher_df = fetch_statcast_details(
        player_type="pitcher",
        player_ids=pitcher_ids,
        windows=windows,
        batch_size=pitcher_batch_size,
    )
    h2h_df = (
        fetch_statcast_matchup_details(
            batter_ids=batter_ids,
            pitcher_ids=pitcher_ids,
            start_date=h2h_start_date,
            end_date=h2h_end_date,
            batter_batch_size=h2h_batter_batch_size,
            pitcher_batch_size=h2h_pitcher_batch_size,
        )
        if h2h_start_date is not None and h2h_end_date is not None and h2h_end_date >= h2h_start_date
        else None
    )
    store = StatcastFeatureStore(batter_df, pitcher_df, h2h_df=h2h_df)
    save_pickle(cache_file, {"schema": MATCHUP_CACHE_SCHEMA, "store": store})
    return store, False, str(cache_file)


def projected_bat_side(bats: str, pitcher_hand: str) -> str:
    bats = (bats or "?").upper()
    pitcher_hand = (pitcher_hand or "?").upper()
    if bats == "S":
        return "R" if pitcher_hand == "L" else "L"
    if bats in {"L", "R"}:
        return bats
    return "R"


def expected_pa_from_lineup_slot(slot: float | None) -> float | None:
    if slot is None:
        return None
    mapping = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.3, 6: 4.1, 7: 3.9, 8: 3.8, 9: 3.7}
    return mapping.get(int(round(slot)), 3.9)


def fetch_park_factor_map(season: int, bat_side: str) -> dict[int, float]:
    params = {
        "batSide": bat_side,
        "condition": "All",
        "rolling": "",
        "stat": "index_Hits",
        "type": "year",
        "year": season,
    }
    try:
        response = requests.get(PARK_FACTORS_URL, params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except Exception:
        return {}
    match = re.search(r"var data = (\[.*?\]);", response.text, flags=re.S)
    if not match:
        return {}
    try:
        raw_rows = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    output = {}
    for row in raw_rows:
        team_id = parse_int(row.get("main_team_id"))
        hit_index = parse_float(row.get("index_hits"))
        if team_id is not None and hit_index is not None:
            output[team_id] = hit_index
    return output


def load_park_factors(season: int) -> dict[str, dict[int, float]]:
    cache_file = cache_path("park_factors", f"{season}.pkl")
    cached = load_pickle(cache_file)
    if isinstance(cached, dict):
        return cached
    factors = {
        "ALL": fetch_park_factor_map(season, ""),
        "L": fetch_park_factor_map(season, "L"),
        "R": fetch_park_factor_map(season, "R"),
    }
    save_pickle(cache_file, factors)
    return factors


def select_park_hit_factor(park_factors: dict[str, dict[int, float]], home_team_id: int, bat_side: str) -> float | None:
    side = bat_side if bat_side in {"L", "R"} else "ALL"
    return park_factors.get(side, {}).get(home_team_id) or park_factors.get("ALL", {}).get(home_team_id)


def load_sprint_speeds(season: int) -> dict[int, float]:
    cache_file = cache_path("sprint_speed", f"{season}.pkl")
    cached = load_pickle(cache_file)
    if isinstance(cached, dict):
        return cached
    try:
        from pybaseball import statcast_sprint_speed

        df = statcast_sprint_speed(season, 1)
    except Exception:
        return {}
    output = {}
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            player_id = parse_int(row.get("player_id"))
            sprint_speed = parse_float(row.get("sprint_speed"))
            if player_id is not None and sprint_speed is not None:
                output[player_id] = sprint_speed
    save_pickle(cache_file, output)
    return output
