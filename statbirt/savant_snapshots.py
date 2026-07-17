from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR
from .utils import parse_float, parse_int


HEADERS = {"User-Agent": "Mozilla/5.0"}
BAT_TRACKING_URL = "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
OAA_URL = "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
DEFAULT_SNAPSHOT_DIR = DATA_DIR / "source_snapshots"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _snapshot_paths(snapshot_dir: Path, source: str, target_date: date) -> tuple[Path, Path]:
    directory = snapshot_dir / source
    return directory / f"{target_date.isoformat()}.csv", directory / f"{target_date.isoformat()}.json"


def _fetch_snapshot(
    *,
    source: str,
    url: str,
    params: dict,
    target_date: date,
    snapshot_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    csv_path, metadata_path = _snapshot_paths(snapshot_dir, source, target_date)
    if csv_path.exists() and metadata_path.exists():
        return pd.read_csv(csv_path, low_memory=False), json.loads(metadata_path.read_text(encoding="utf-8"))
    # Leaderboards are current-season snapshots and must never be substituted for a historical as-of date.
    if target_date != date.today():
        return pd.DataFrame(), {
            "source": source,
            "target_date": target_date.isoformat(),
            "backfill_safety": "prospective_only_missing_archive",
        }
    response = requests.get(url, params=params, headers=HEADERS, timeout=60)
    response.raise_for_status()
    content = response.content.decode("utf-8-sig")
    if content.lstrip().startswith("<"):
        raise ValueError(f"Baseball Savant {source} returned HTML instead of CSV.")
    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    observed_at = _now_utc()
    metadata = {
        "snapshot_id": f"{source}-{target_date.isoformat()}-{source_hash[:12]}",
        "source": source,
        "url": response.url,
        "target_date": target_date.isoformat(),
        "observed_at_utc": observed_at,
        "source_max_game_date": (target_date - timedelta(days=1)).isoformat(),
        "source_hash": source_hash,
        "backfill_safety": "prospective_snapshot_only",
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(content, encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return pd.read_csv(io.StringIO(content), low_memory=False), metadata


def load_bat_tracking_snapshot(
    target_date: date,
    *,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> tuple[dict[int, dict[str, float | int | str | None]], dict]:
    frame, metadata = _fetch_snapshot(
        source="bat_tracking",
        url=BAT_TRACKING_URL,
        params={
            "gameType": "Regular",
            "seasonStart": target_date.year,
            "seasonEnd": target_date.year,
            "type": "batter",
            "minSwings": 1,
            "csv": "true",
        },
        target_date=target_date,
        snapshot_dir=snapshot_dir,
    )
    output = {}
    for row in frame.to_dict("records"):
        player_id = parse_int(row.get("id"))
        if player_id is None:
            continue
        competitive_swings = parse_int(row.get("swings_competitive"))
        contact = parse_int(row.get("contact"))
        output[player_id] = {
            "competitive_swings": competitive_swings,
            "competitive_contact_rate": (
                contact / competitive_swings if contact is not None and competitive_swings else None
            ),
            "avg_bat_speed": parse_float(row.get("avg_bat_speed")),
            "avg_swing_length": parse_float(row.get("swing_length")),
            "squared_up_per_contact": parse_float(row.get("squared_up_per_bat_contact")),
            "blast_per_contact": parse_float(row.get("blast_per_bat_contact")),
            "snapshot_id": metadata.get("snapshot_id"),
            "source_hash": metadata.get("source_hash"),
        }
    return output, metadata


def _team_id_for_display_name(display_name: str, team_metadata: dict[int, dict]) -> int | None:
    needle = str(display_name or "").strip().lower()
    if not needle:
        return None
    aliases = {"d-backs": "arizona diamondbacks"}
    needle = aliases.get(needle, needle)
    matches = []
    for team_id, values in team_metadata.items():
        full_name = str(values.get("name") or "").strip().lower()
        abbr = str(values.get("abbr") or "").strip().lower()
        if needle == full_name or needle == abbr or full_name.endswith(f" {needle}"):
            matches.append(team_id)
    return matches[0] if len(matches) == 1 else None


def load_oaa_snapshot(
    target_date: date,
    team_metadata: dict[int, dict],
    *,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> tuple[dict[int, dict[str, float | str | None]], dict]:
    frame, metadata = _fetch_snapshot(
        source="outs_above_average",
        url=OAA_URL,
        params={
            "type": "Fielder",
            "startYear": target_date.year,
            "endYear": target_date.year,
            "split": "yes",
            "min": 0,
            "csv": "true",
        },
        target_date=target_date,
        snapshot_dir=snapshot_dir,
    )
    totals: dict[int, dict[str, float | str | None]] = {}
    infield_positions = {"1B", "2B", "3B", "SS"}
    outfield_positions = {"LF", "CF", "RF", "OF"}
    for row in frame.to_dict("records"):
        team_id = _team_id_for_display_name(str(row.get("display_team_name") or ""), team_metadata)
        value = parse_float(row.get("outs_above_average"))
        if team_id is None or value is None:
            continue
        record = totals.setdefault(
            team_id,
            {
                "team_oaa": 0.0,
                "infield_oaa": 0.0,
                "outfield_oaa": 0.0,
                "snapshot_id": metadata.get("snapshot_id"),
                "source_hash": metadata.get("source_hash"),
            },
        )
        record["team_oaa"] = float(record["team_oaa"] or 0.0) + value
        position = str(row.get("primary_pos_formatted") or "").strip().upper()
        if position in infield_positions:
            record["infield_oaa"] = float(record["infield_oaa"] or 0.0) + value
        elif position in outfield_positions:
            record["outfield_oaa"] = float(record["outfield_oaa"] or 0.0) + value
    return totals, metadata
