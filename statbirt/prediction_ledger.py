from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Iterable, Iterator

from .config import DATA_DIR, DEFAULT_OUTPUT_CSV


DEFAULT_LEDGER_DIR = DATA_DIR / "decision_ledger"
DEFAULT_SNAPSHOTS_CSV = DEFAULT_LEDGER_DIR / "snapshots.csv"
DEFAULT_RESULTS_CSV = DEFAULT_LEDGER_DIR / "results.csv"
DEFAULT_PRODUCTION_PREDICTIONS_CSV = DATA_DIR / "model_predictions.csv"
DEFAULT_SHADOW_PREDICTIONS_CSV = DATA_DIR / "learned_shadow_predictions.csv"

SCHEMA_VERSION = "1"

SNAPSHOT_FIELDS = [
    "schema_version",
    "run_id",
    "saved_at",
    "cutoff_at",
    "target_date",
    "candidate_key",
    "candidate_pool_source",
    "lineup_provenance",
    "player",
    "player_id",
    "team",
    "opponent",
    "game_pk",
    "game_start_time_utc",
    "model_version",
    "production_model_trained_at",
    "shadow_model_version",
    "shadow_model_trained_at",
    "shadow_scored_at",
    "feature_snapshot_id",
    "feature_hash",
    "feature_source_hash",
    "lineup_observed_at_utc",
    "raw_probability",
    "calibrated_probability",
    "appearance_probability",
    "combined_probability",
    "hit_given_appearance_probability",
    "combined_probability_raw",
    "combined_probability_calibrated",
    "production_rank",
    "shadow_rank",
    "shadow_top5_rank",
    "stuff_preference_rank",
    "snapshot_hash",
]

RESULT_FIELDS = [
    "schema_version",
    "run_id",
    "candidate_key",
    "snapshot_hash",
    "target_date",
    "player",
    "player_id",
    "game_pk",
    "result_hit",
    "result_hits",
    "result_ab",
    "result_pa",
    "result_status",
    "result_updated_at",
    "notes",
    "synced_at",
]

RESULT_VALUE_FIELDS = [
    "result_hit",
    "result_hits",
    "result_ab",
    "result_pa",
    "result_status",
    "result_updated_at",
    "notes",
]

RESULT_STATUS_VALUES = {
    "",
    "final",
    "pending",
    "postponed",
    "no_appearance",
    "unresolved",
}

RAW_PROBABILITY_FIELDS = (
    "raw_probability",
    "raw_hit_probability",
    "learned_hit_probability",
    "hit_probability",
    "probability",
)
CALIBRATED_PROBABILITY_FIELDS = (
    "calibrated_probability",
    "calibrated_hit_probability",
)
APPEARANCE_PROBABILITY_FIELDS = (
    "appearance_probability",
    "appearance_prob",
)
COMBINED_PROBABILITY_FIELDS = (
    "combined_probability",
    "combined_hit_probability",
    "decision_probability",
)
PRODUCTION_RANK_FIELDS = ("production_rank", "learned_rank", "rank")
SHADOW_RANK_FIELDS = ("shadow_rank", "learned_rank", "rank")
MODEL_VERSION_FIELDS = ("model_version", "production_model_version")
SHADOW_MODEL_VERSION_FIELDS = ("shadow_model_version", "model_version")
SHADOW_PRODUCTION_PROBABILITY_FIELDS = ("production_probability", *RAW_PROBABILITY_FIELDS)
HIT_GIVEN_APPEARANCE_FIELDS = ("hit_given_appearance_probability", "conditional_hit_probability")
COMBINED_RAW_FIELDS = ("combined_probability_raw", "raw_combined_probability")
COMBINED_CALIBRATED_FIELDS = (
    "combined_probability_calibrated",
    "calibrated_combined_probability",
)


class LedgerError(ValueError):
    """Raised when a ledger operation would create ambiguous or invalid history."""


class LedgerConflictError(LedgerError):
    """Raised when an existing immutable run does not match a rerun."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _first(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = _text(row.get(name))
        if value:
            return value
    return ""


def _canonical_timestamp(value: str, field: str) -> str:
    text = _text(value)
    if not text:
        raise LedgerError(f"{field} must not be blank.")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerError(f"{field} must be an ISO-8601 timestamp: {text!r}.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LedgerError(f"{field} must include a timezone offset.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _timestamp_value(value: str, field: str) -> datetime:
    canonical = _canonical_timestamp(value, field)
    return datetime.fromisoformat(canonical.replace("Z", "+00:00"))


def _canonical_date(value: str, field: str = "target_date") -> str:
    text = _text(value)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise LedgerError(f"{field} must be an ISO date: {text!r}.") from exc


def _canonical_probability(value: str, field: str, *, required: bool = False) -> str:
    text = _text(value)
    if not text:
        if required:
            raise LedgerError(f"{field} must not be blank.")
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise LedgerError(f"{field} must be numeric: {text!r}.") from exc
    if not number.is_finite() or number < 0 or number > 1:
        raise LedgerError(f"{field} must be between 0 and 1: {text!r}.")
    return format(number.normalize(), "f")


def _canonical_rank(value: str, field: str, *, required: bool = False) -> str:
    text = _text(value)
    if not text:
        if required:
            raise LedgerError(f"{field} must not be blank.")
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise LedgerError(f"{field} must be a positive integer: {text!r}.") from exc
    if not number.is_finite() or number != number.to_integral_value() or number < 1:
        raise LedgerError(f"{field} must be a positive integer: {text!r}.")
    return str(int(number))


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _text(value).lower()).strip("-")


def _row_date(row: dict[str, str], default: str = "") -> str:
    return _text(row.get("date") or row.get("target_date") or default)


def _identity_tokens(row: dict[str, str], default_date: str = "") -> tuple[str, ...]:
    tokens: list[str] = []
    explicit = _text(row.get("candidate_key"))
    if explicit:
        tokens.append(f"key:{explicit}")

    row_date = _row_date(row, default_date)
    player_id = _text(row.get("player_id"))
    game_pk = _text(row.get("game_pk"))
    if row_date and player_id and game_pk:
        tokens.append(f"game:{row_date}|{player_id}|{game_pk}")
    elif row_date and _text(row.get("player")):
        tokens.append(
            "fallback:"
            + "|".join(
                (
                    row_date,
                    _normalized_name(_text(row.get("player"))),
                    _text(row.get("team")).upper(),
                    _text(row.get("opponent")).upper(),
                )
            )
        )
    return tuple(tokens)


def make_candidate_key(row: dict[str, str], target_date: str = "") -> str:
    """Return the stable, human-readable identity stored in snapshot rows."""
    explicit = _text(row.get("candidate_key"))
    if explicit:
        if "\n" in explicit or "\r" in explicit:
            raise LedgerError("candidate_key must not contain a newline.")
        return explicit

    row_date = _row_date(row, target_date)
    player_id = _text(row.get("player_id"))
    game_pk = _text(row.get("game_pk"))
    if row_date and player_id and game_pk:
        return f"{row_date}|{player_id}|{game_pk}"

    player = _normalized_name(_text(row.get("player")))
    if row_date and player:
        return "fallback|" + "|".join(
            (row_date, player, _text(row.get("team")).upper(), _text(row.get("opponent")).upper())
        )
    raise LedgerError("Candidate rows need candidate_key or date/player identity fields.")


def _read_csv(path: Path, *, allow_missing: bool = False) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        if allow_missing:
            return [], []
        raise LedgerError(f"CSV file does not exist: {path}")
    if path.stat().st_size == 0:
        if allow_missing:
            return [], []
        raise LedgerError(f"CSV file is empty: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise LedgerError(f"CSV file has no header: {path}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise LedgerError(f"Malformed CSV row at {path}:{line_number}.")
            rows.append({field: _text(raw.get(field)) for field in fieldnames})
    return rows, fieldnames


def _require_schema(path: Path, fieldnames: list[str], expected: list[str]) -> None:
    if fieldnames != expected:
        missing = [field for field in expected if field not in fieldnames]
        extra = [field for field in fieldnames if field not in expected]
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        if not details:
            details.append("columns are out of order")
        raise LedgerError(f"Unsupported ledger schema in {path}: {'; '.join(details)}.")


@contextmanager
def _file_lock(path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise LedgerError(f"Timed out waiting for ledger lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _text(row.get(field)) for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _append_snapshots(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: _text(row.get(field)) for field in SNAPSHOT_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _snapshot_hash(row: dict[str, str]) -> str:
    payload = {field: _text(row.get(field)) for field in SNAPSHOT_FIELDS if field != "snapshot_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_feature_hash(row: dict[str, str]) -> str:
    payload = {
        field: _text(value)
        for field, value in row.items()
        if not field.startswith("result_") and field not in {"notes"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_date(rows: list[dict[str, str]], requested: str | None) -> str:
    dates = sorted({_row_date(row) for row in rows if _row_date(row)})
    value = _text(requested or "latest")
    if value.lower() == "latest":
        if not dates:
            raise LedgerError("Candidate CSV has no dated rows.")
        value = dates[-1]
    target = _canonical_date(value)
    if target not in dates:
        raise LedgerError(f"Candidate CSV has no rows for target date {target}.")
    return target


def _rows_for_date(rows: list[dict[str, str]], target_date: str) -> list[dict[str, str]]:
    if not target_date:
        return list(rows)
    return [row for row in rows if not _row_date(row) or _row_date(row) == target_date]


def _index_rows(rows: list[dict[str, str]], target_date: str, label: str) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in _rows_for_date(rows, target_date):
        tokens = _identity_tokens(row, target_date)
        if not tokens:
            raise LedgerError(f"{label} row has no usable candidate identity.")
        for token in tokens:
            previous = index.get(token)
            if previous is not None and previous is not row:
                raise LedgerError(f"{label} rows are not unique for identity {token!r}.")
            index[token] = row
    return index


def _match_row(
    candidate: dict[str, str],
    index: dict[str, dict[str, str]],
    target_date: str,
) -> dict[str, str] | None:
    matches = {id(index[token]): index[token] for token in _identity_tokens(candidate, target_date) if token in index}
    if len(matches) > 1:
        raise LedgerError(f"Multiple prediction rows match candidate {make_candidate_key(candidate, target_date)!r}.")
    return next(iter(matches.values()), None)


def _lineup_provenance(
    candidate: dict[str, str],
    shadow: dict[str, str],
    production: dict[str, str],
    override: str | None,
) -> str:
    if _text(override):
        return _text(override)
    explicit = (
        _first(candidate, ("lineup_provenance", "lineup_source"))
        or _first(shadow, ("lineup_provenance", "lineup_source"))
        or _first(production, ("lineup_provenance", "lineup_source"))
    )
    if explicit:
        return explicit
    confirmed = _text(candidate.get("confirmed_lineup")).lower()
    if confirmed in {"y", "yes", "true", "1"}:
        return "official"
    return "unconfirmed"


def _validate_pregame_times(
    candidates: list[dict[str, str]], cutoff_at: str, saved_at: str, target_date: str
) -> None:
    cutoff = _timestamp_value(cutoff_at, "cutoff_at")
    saved = _timestamp_value(saved_at, "saved_at")
    if cutoff > saved:
        raise LedgerError("cutoff_at must not be later than saved_at.")
    for candidate in candidates:
        start_text = _first(candidate, ("game_start_time_utc", "game_start_at", "first_pitch_at"))
        if not start_text:
            continue
        start = _timestamp_value(start_text, "game_start_time_utc")
        key = make_candidate_key(candidate, target_date)
        if cutoff >= start:
            raise LedgerError(f"cutoff_at is not pregame for candidate {key!r}.")
        if saved >= start:
            raise LedgerError(f"saved_at is not pregame for candidate {key!r}.")


def _rank_sort(row: dict[str, str]) -> tuple[int, str]:
    rank = _text(row.get("production_rank"))
    return (int(rank) if rank else 2_147_483_647, _text(row.get("candidate_key")))


def _build_snapshot_rows(
    *,
    run_id: str,
    candidates: list[dict[str, str]],
    production_rows: list[dict[str, str]],
    shadow_rows: list[dict[str, str]],
    target_date: str,
    cutoff_at: str,
    saved_at: str,
    candidate_pool_source_override: str,
    candidate_pool_source_default: str,
    lineup_provenance: str | None,
    shadow_supplied: bool,
) -> list[dict[str, str]]:
    candidate_index = _index_rows(candidates, target_date, "Candidate")
    production_index = _index_rows(production_rows, target_date, "Production prediction")
    shadow_index = _index_rows(shadow_rows, target_date, "Shadow prediction") if shadow_supplied else {}

    selected_candidates = [row for row in candidates if _row_date(row) == target_date]
    if not selected_candidates:
        raise LedgerError(f"No candidate rows matched target date {target_date}.")

    candidate_keys: set[str] = set()
    used_production_rows: set[int] = set()
    used_shadow_rows: set[int] = set()
    snapshots: list[dict[str, str]] = []
    for candidate in selected_candidates:
        candidate_key = make_candidate_key(candidate, target_date)
        if candidate_key in candidate_keys:
            raise LedgerError(f"Candidate keys are not unique: {candidate_key!r}.")
        candidate_keys.add(candidate_key)

        production = _match_row(candidate, production_index, target_date)
        if production is None:
            raise LedgerError(f"Missing production prediction for candidate {candidate_key!r}.")
        used_production_rows.add(id(production))

        shadow = _match_row(candidate, shadow_index, target_date) if shadow_supplied else None
        if shadow is not None:
            used_shadow_rows.add(id(shadow))

        production_rank = _canonical_rank(
            _first(production, PRODUCTION_RANK_FIELDS), "production_rank", required=True
        )
        shadow_rank_value = _first(shadow or production, SHADOW_RANK_FIELDS if shadow else ("shadow_rank",))
        shadow_rank = _canonical_rank(
            shadow_rank_value,
            "shadow_rank",
            required=shadow_supplied and shadow is not None,
        )
        shadow_top5_rank = _canonical_rank(
            _text((shadow or {}).get("shadow_top5_rank")),
            "shadow_top5_rank",
        )
        stuff_preference_rank = _canonical_rank(
            _text((shadow or {}).get("stuff_preference_rank")),
            "stuff_preference_rank",
        )

        raw_probability = _first(production, RAW_PROBABILITY_FIELDS) or _first(
            shadow or {}, SHADOW_PRODUCTION_PROBABILITY_FIELDS
        )
        combined_raw = _first(shadow or {}, COMBINED_RAW_FIELDS) or _first(
            production, COMBINED_RAW_FIELDS
        )
        combined_calibrated = _first(shadow or {}, COMBINED_CALIBRATED_FIELDS) or _first(
            production, COMBINED_CALIBRATED_FIELDS
        )
        calibrated_probability = _first(production, CALIBRATED_PROBABILITY_FIELDS) or combined_calibrated
        combined_probability = (
            _first(production, COMBINED_PROBABILITY_FIELDS)
            or _first(shadow or {}, COMBINED_PROBABILITY_FIELDS)
            or combined_calibrated
            or combined_raw
        )
        pool_source = (
            candidate_pool_source_override
            or _text(candidate.get("candidate_pool_source"))
            or _text((shadow or {}).get("candidate_pool_source"))
            or _text(production.get("candidate_pool_source"))
            or candidate_pool_source_default
        )

        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "saved_at": saved_at,
            "cutoff_at": cutoff_at,
            "target_date": target_date,
            "candidate_key": candidate_key,
            "candidate_pool_source": pool_source,
            "lineup_provenance": _lineup_provenance(
                candidate, shadow or {}, production, lineup_provenance
            ),
            "player": _text(candidate.get("player") or production.get("player")),
            "player_id": _text(candidate.get("player_id") or production.get("player_id")),
            "team": _text(candidate.get("team") or production.get("team")),
            "opponent": _text(candidate.get("opponent") or production.get("opponent")),
            "game_pk": _text(candidate.get("game_pk") or production.get("game_pk")),
            "game_start_time_utc": _text(candidate.get("game_start_time_utc")),
            "model_version": _first(production, MODEL_VERSION_FIELDS),
            "production_model_trained_at": _text(production.get("model_trained_at")),
            "shadow_model_version": _first(shadow or {}, SHADOW_MODEL_VERSION_FIELDS),
            "shadow_model_trained_at": _text((shadow or {}).get("model_trained_at")),
            "shadow_scored_at": _text((shadow or {}).get("scored_at")),
            "feature_snapshot_id": _text(
                candidate.get("feature_snapshot_id") or (shadow or {}).get("feature_snapshot_id")
            ),
            "feature_hash": _candidate_feature_hash(candidate),
            "feature_source_hash": _text(
                candidate.get("feature_source_hash") or (shadow or {}).get("feature_source_hash")
            ),
            "lineup_observed_at_utc": _text(candidate.get("lineup_observed_at_utc")),
            "raw_probability": _canonical_probability(
                raw_probability, "raw_probability", required=True
            ),
            "calibrated_probability": _canonical_probability(
                calibrated_probability, "calibrated_probability", required=True
            ),
            "appearance_probability": _canonical_probability(
                _first(production, APPEARANCE_PROBABILITY_FIELDS)
                or _first(shadow or {}, APPEARANCE_PROBABILITY_FIELDS),
                "appearance_probability",
                required=True,
            ),
            "combined_probability": _canonical_probability(
                combined_probability, "combined_probability", required=True
            ),
            "hit_given_appearance_probability": _canonical_probability(
                _first(shadow or {}, HIT_GIVEN_APPEARANCE_FIELDS)
                or _first(production, HIT_GIVEN_APPEARANCE_FIELDS),
                "hit_given_appearance_probability",
            ),
            "combined_probability_raw": _canonical_probability(
                combined_raw, "combined_probability_raw"
            ),
            "combined_probability_calibrated": _canonical_probability(
                combined_calibrated, "combined_probability_calibrated"
            ),
            "production_rank": production_rank,
            "shadow_rank": shadow_rank,
            "shadow_top5_rank": shadow_top5_rank,
            "stuff_preference_rank": stuff_preference_rank,
            "snapshot_hash": "",
        }
        if not row["model_version"]:
            raise LedgerError(f"Missing model_version for candidate {candidate_key!r}.")
        if shadow_supplied and not row["shadow_model_version"]:
            raise LedgerError(f"Missing shadow model_version for candidate {candidate_key!r}.")
        if not row["candidate_pool_source"]:
            raise LedgerError(f"Missing candidate_pool_source for candidate {candidate_key!r}.")
        if not row["lineup_provenance"]:
            raise LedgerError(f"Missing lineup_provenance for candidate {candidate_key!r}.")
        game_start = row["game_start_time_utc"]
        if game_start:
            start_value = _timestamp_value(game_start, "game_start_time_utc")
            for timestamp_field in (
                "production_model_trained_at",
                "shadow_model_trained_at",
                "shadow_scored_at",
                "lineup_observed_at_utc",
            ):
                timestamp = row[timestamp_field]
                if timestamp and _timestamp_value(timestamp, timestamp_field) >= start_value:
                    raise LedgerError(
                        f"{timestamp_field} is not pregame for candidate {candidate_key!r}."
                    )
        row["snapshot_hash"] = _snapshot_hash(row)
        snapshots.append(row)

    production_input = {id(row) for row in _rows_for_date(production_rows, target_date)}
    unused_production = production_input - used_production_rows
    if unused_production:
        raise LedgerError(f"Found {len(unused_production)} production prediction row(s) outside the candidate pool.")
    if shadow_supplied:
        shadow_input = {id(row) for row in _rows_for_date(shadow_rows, target_date)}
        unused_shadow = shadow_input - used_shadow_rows
        if unused_shadow:
            raise LedgerError(f"Found {len(unused_shadow)} shadow prediction row(s) outside the candidate pool.")

    for rank_field in ("production_rank", "shadow_rank"):
        ranks = [_text(row.get(rank_field)) for row in snapshots if _text(row.get(rank_field))]
        if len(ranks) != len(set(ranks)):
            raise LedgerError(f"Snapshot has duplicate {rank_field} values.")
    model_versions = {_text(row.get("model_version")) for row in snapshots}
    if len(model_versions) != 1:
        raise LedgerError("Production predictions contain multiple model versions for one run.")
    del candidate_index  # Duplicate candidate identities were checked while constructing the index.
    return sorted(snapshots, key=_rank_sort)


def create_snapshot(
    *,
    run_id: str,
    candidates_csv: str | Path = DEFAULT_OUTPUT_CSV,
    production_predictions_csv: str | Path = DEFAULT_PRODUCTION_PREDICTIONS_CSV,
    shadow_predictions_csv: str | Path | None = DEFAULT_SHADOW_PREDICTIONS_CSV,
    snapshots_csv: str | Path = DEFAULT_SNAPSHOTS_CSV,
    target_date: str | None = "latest",
    cutoff_at: str | None = None,
    saved_at: str | None = None,
    candidate_pool_source: str | None = None,
    lineup_provenance: str | None = None,
) -> dict[str, object]:
    """Append a frozen decision board, or return an idempotent no-op for the same run."""
    run_id = _text(run_id)
    if not run_id or "\n" in run_id or "\r" in run_id:
        raise LedgerError("run_id must be non-empty and must not contain a newline.")

    candidates_path = Path(candidates_csv)
    production_path = Path(production_predictions_csv)
    shadow_path = Path(shadow_predictions_csv) if shadow_predictions_csv is not None else None
    snapshots_path = Path(snapshots_csv)

    candidates, _ = _read_csv(candidates_path)
    production_rows, _ = _read_csv(production_path)
    shadow_rows, _ = _read_csv(shadow_path) if shadow_path is not None else ([], [])
    resolved_target = _target_date(candidates, target_date)
    pool_source_override = _text(candidate_pool_source)
    pool_source_default = str(candidates_path.resolve())

    with _file_lock(snapshots_path):
        existing, existing_fields = _read_csv(snapshots_path, allow_missing=True)
        if existing_fields:
            _require_schema(snapshots_path, existing_fields, SNAPSHOT_FIELDS)
        existing_run = [row for row in existing if _text(row.get("run_id")) == run_id]
        if existing_run:
            existing_saved = {_text(row.get("saved_at")) for row in existing_run}
            existing_cutoff = {_text(row.get("cutoff_at")) for row in existing_run}
            if len(existing_saved) != 1 or len(existing_cutoff) != 1:
                raise LedgerConflictError(f"Existing run {run_id!r} has inconsistent timestamps.")
            if saved_at is None:
                saved_at = next(iter(existing_saved))
            if cutoff_at is None:
                cutoff_at = next(iter(existing_cutoff))

        resolved_saved = _canonical_timestamp(saved_at or _now_utc(), "saved_at")
        resolved_cutoff = _canonical_timestamp(cutoff_at or resolved_saved, "cutoff_at")
        target_candidates = [row for row in candidates if _row_date(row) == resolved_target]
        _validate_pregame_times(target_candidates, resolved_cutoff, resolved_saved, resolved_target)

        new_rows = _build_snapshot_rows(
            run_id=run_id,
            candidates=candidates,
            production_rows=production_rows,
            shadow_rows=shadow_rows,
            target_date=resolved_target,
            cutoff_at=resolved_cutoff,
            saved_at=resolved_saved,
            candidate_pool_source_override=pool_source_override,
            candidate_pool_source_default=pool_source_default,
            lineup_provenance=lineup_provenance,
            shadow_supplied=shadow_path is not None,
        )

        if existing_run:
            existing_sorted = sorted(existing_run, key=_rank_sort)
            if existing_sorted != new_rows:
                raise LedgerConflictError(
                    f"Run {run_id!r} already exists with a different immutable snapshot. Use a new run_id."
                )
            return {
                "run_id": run_id,
                "target_date": resolved_target,
                "snapshot_rows": len(new_rows),
                "appended_rows": 0,
                "idempotent": True,
                "snapshots_csv": str(snapshots_path.resolve()),
            }

        _append_snapshots(snapshots_path, new_rows)
        return {
            "run_id": run_id,
            "target_date": resolved_target,
            "snapshot_rows": len(new_rows),
            "appended_rows": len(new_rows),
            "idempotent": False,
            "snapshots_csv": str(snapshots_path.resolve()),
        }


def _normalize_result_status(row: dict[str, str]) -> str:
    status = _text(row.get("result_status")).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "game_over": "final",
        "completed_early": "final",
        "completed": "final",
        "final": "final",
        "in_progress": "pending",
        "scheduled": "pending",
        "pre_game": "pending",
        "pregame": "pending",
        "no_appearance": "no_appearance",
        "did_not_appear": "no_appearance",
    }
    normalized = aliases.get(status, status)
    if not normalized and _text(row.get("result_hit")) in {"0", "1"}:
        return "final"
    return normalized


def _result_identity(row: dict[str, str]) -> tuple[str, str]:
    return (_text(row.get("run_id")), _text(row.get("candidate_key")))


def _result_base(snapshot: dict[str, str]) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": _text(snapshot.get("run_id")),
        "candidate_key": _text(snapshot.get("candidate_key")),
        "snapshot_hash": _text(snapshot.get("snapshot_hash")),
        "target_date": _text(snapshot.get("target_date")),
        "player": _text(snapshot.get("player")),
        "player_id": _text(snapshot.get("player_id")),
        "game_pk": _text(snapshot.get("game_pk")),
        **{field: "" for field in RESULT_VALUE_FIELDS},
        "synced_at": "",
    }


def _results_sort(row: dict[str, str]) -> tuple[str, str, str]:
    return (_text(row.get("target_date")), _text(row.get("run_id")), _text(row.get("candidate_key")))


def sync_results(
    *,
    candidates_csv: str | Path = DEFAULT_OUTPUT_CSV,
    snapshots_csv: str | Path = DEFAULT_SNAPSHOTS_CSV,
    results_csv: str | Path = DEFAULT_RESULTS_CSV,
    synced_at: str | None = None,
) -> dict[str, object]:
    """Upsert mutable outcomes without changing any immutable snapshot row."""
    candidates_path = Path(candidates_csv)
    snapshots_path = Path(snapshots_csv)
    results_path = Path(results_csv)
    synced = _canonical_timestamp(synced_at or _now_utc(), "synced_at")

    candidates, _ = _read_csv(candidates_path)
    snapshots, snapshot_fields = _read_csv(snapshots_path)
    _require_schema(snapshots_path, snapshot_fields, SNAPSHOT_FIELDS)
    candidate_index = _index_rows(candidates, "", "Result source candidate")

    with _file_lock(results_path):
        existing, existing_fields = _read_csv(results_path, allow_missing=True)
        if existing_fields:
            _require_schema(results_path, existing_fields, RESULT_FIELDS)
        existing_by_key: dict[tuple[str, str], dict[str, str]] = {}
        for row in existing:
            key = _result_identity(row)
            if not all(key):
                raise LedgerError("Existing result rows require run_id and candidate_key.")
            if key in existing_by_key:
                raise LedgerError(f"Existing results contain duplicate identity {key!r}.")
            existing_by_key[key] = row

        inserted = 0
        updated = 0
        unchanged = 0
        unmatched = 0
        output_by_key = dict(existing_by_key)
        for snapshot in snapshots:
            key = _result_identity(snapshot)
            if not all(key):
                raise LedgerError("Snapshot rows require run_id and candidate_key before results can sync.")
            existing_row = existing_by_key.get(key)
            desired = _result_base(snapshot)
            if existing_row:
                for field in RESULT_VALUE_FIELDS:
                    desired[field] = _text(existing_row.get(field))

            candidate = _match_row(snapshot, candidate_index, _text(snapshot.get("target_date")))
            if candidate is None:
                unmatched += 1
            else:
                incoming = {field: _text(candidate.get(field)) for field in RESULT_VALUE_FIELDS}
                incoming["result_status"] = _normalize_result_status(candidate)
                for field, value in incoming.items():
                    if value or existing_row is None:
                        desired[field] = value

            comparison_fields = [field for field in RESULT_FIELDS if field != "synced_at"]
            changed = existing_row is None or any(
                _text(existing_row.get(field)) != _text(desired.get(field)) for field in comparison_fields
            )
            if existing_row is None:
                inserted += 1
                desired["synced_at"] = synced
            elif changed:
                updated += 1
                desired["synced_at"] = synced
            else:
                unchanged += 1
                desired["synced_at"] = _text(existing_row.get("synced_at")) or synced
            output_by_key[key] = desired

        output = sorted(output_by_key.values(), key=_results_sort)
        if output != existing or not results_path.exists():
            _write_csv_atomic(results_path, RESULT_FIELDS, output)

    return {
        "snapshot_rows": len(snapshots),
        "result_rows": len(output),
        "inserted_rows": inserted,
        "updated_rows": updated,
        "unchanged_rows": unchanged,
        "unmatched_snapshot_rows": unmatched,
        "results_csv": str(results_path.resolve()),
    }


def audit_ledger(
    *,
    snapshots_csv: str | Path = DEFAULT_SNAPSHOTS_CSV,
    results_csv: str | Path = DEFAULT_RESULTS_CSV,
) -> dict[str, object]:
    """Audit schema, immutable row digests, identities, ranks, and result foreign keys."""
    snapshots_path = Path(snapshots_csv)
    results_path = Path(results_csv)
    errors: list[str] = []
    warnings: list[str] = []

    try:
        snapshots, snapshot_fields = _read_csv(snapshots_path)
        _require_schema(snapshots_path, snapshot_fields, SNAPSHOT_FIELDS)
    except LedgerError as exc:
        snapshots = []
        errors.append(str(exc))

    snapshot_by_key: dict[tuple[str, str], dict[str, str]] = {}
    runs: dict[str, list[dict[str, str]]] = {}
    for line_number, row in enumerate(snapshots, start=2):
        prefix = f"snapshot row {line_number}"
        identity = _result_identity(row)
        if not all(identity):
            errors.append(f"{prefix}: run_id and candidate_key are required.")
        elif identity in snapshot_by_key:
            errors.append(f"{prefix}: duplicate snapshot identity {identity!r}.")
        else:
            snapshot_by_key[identity] = row
        runs.setdefault(_text(row.get("run_id")), []).append(row)

        if _text(row.get("schema_version")) != SCHEMA_VERSION:
            errors.append(f"{prefix}: unsupported schema_version {_text(row.get('schema_version'))!r}.")
        if _snapshot_hash(row) != _text(row.get("snapshot_hash")):
            errors.append(f"{prefix}: snapshot_hash does not match immutable content.")
        for field in ("saved_at", "cutoff_at"):
            try:
                _canonical_timestamp(_text(row.get(field)), field)
            except LedgerError as exc:
                errors.append(f"{prefix}: {exc}")
        try:
            _canonical_date(_text(row.get("target_date")))
        except LedgerError as exc:
            errors.append(f"{prefix}: {exc}")
        for field in (
            "raw_probability",
            "calibrated_probability",
            "appearance_probability",
            "combined_probability",
            "hit_given_appearance_probability",
            "combined_probability_raw",
            "combined_probability_calibrated",
        ):
            try:
                _canonical_probability(
                    _text(row.get(field)),
                    field,
                    required=field
                    in {
                        "raw_probability",
                        "calibrated_probability",
                        "appearance_probability",
                        "combined_probability",
                    },
                )
            except LedgerError as exc:
                errors.append(f"{prefix}: {exc}")
        for field in ("production_rank", "shadow_rank"):
            try:
                _canonical_rank(_text(row.get(field)), field, required=True)
            except LedgerError as exc:
                errors.append(f"{prefix}: {exc}")
        for field in ("candidate_pool_source", "lineup_provenance", "model_version"):
            if not _text(row.get(field)):
                errors.append(f"{prefix}: {field} is required.")
        if not re.fullmatch(r"[0-9a-f]{64}", _text(row.get("feature_hash"))):
            errors.append(f"{prefix}: feature_hash must be a SHA-256 digest.")
        if _text(row.get("shadow_rank")) and not _text(row.get("shadow_model_version")):
            errors.append(f"{prefix}: shadow_model_version is required with shadow_rank.")

    for run_id, rows in runs.items():
        if not run_id:
            continue
        for field in ("saved_at", "cutoff_at", "target_date", "model_version"):
            values = {_text(row.get(field)) for row in rows}
            if len(values) != 1:
                errors.append(f"run {run_id!r}: inconsistent {field} values.")
        for rank_field in ("production_rank", "shadow_rank"):
            ranks = [_text(row.get(rank_field)) for row in rows if _text(row.get(rank_field))]
            if len(ranks) != len(set(ranks)):
                errors.append(f"run {run_id!r}: duplicate {rank_field} values.")

    try:
        results, result_fields = _read_csv(results_path, allow_missing=True)
        if result_fields:
            _require_schema(results_path, result_fields, RESULT_FIELDS)
    except LedgerError as exc:
        results = []
        errors.append(str(exc))

    result_keys: set[tuple[str, str]] = set()
    resolved_results = 0
    for line_number, row in enumerate(results, start=2):
        prefix = f"result row {line_number}"
        key = _result_identity(row)
        if not all(key):
            errors.append(f"{prefix}: run_id and candidate_key are required.")
        elif key in result_keys:
            errors.append(f"{prefix}: duplicate result identity {key!r}.")
        result_keys.add(key)
        snapshot = snapshot_by_key.get(key)
        if snapshot is None:
            errors.append(f"{prefix}: result has no matching snapshot {key!r}.")
        elif _text(row.get("snapshot_hash")) != _text(snapshot.get("snapshot_hash")):
            errors.append(f"{prefix}: snapshot_hash does not match its snapshot row.")
        status = _text(row.get("result_status"))
        if status not in RESULT_STATUS_VALUES:
            errors.append(f"{prefix}: unknown result_status {status!r}.")
        if status == "final":
            resolved_results += 1
        if _text(row.get("schema_version")) != SCHEMA_VERSION:
            errors.append(f"{prefix}: unsupported schema_version {_text(row.get('schema_version'))!r}.")

    missing_result_rows = len(set(snapshot_by_key) - result_keys)
    if not results_path.exists():
        warnings.append(f"Results ledger does not exist yet: {results_path}")
    elif missing_result_rows:
        warnings.append(f"{missing_result_rows} snapshot row(s) do not yet have a result row.")

    return {
        "ok": not errors,
        "snapshot_rows": len(snapshots),
        "runs": len([run_id for run_id in runs if run_id]),
        "result_rows": len(results),
        "resolved_result_rows": resolved_results,
        "missing_result_rows": missing_result_rows,
        "errors": errors,
        "warnings": warnings,
        "snapshots_csv": str(snapshots_path.resolve()),
        "results_csv": str(results_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain the immutable learned-model decision ledger.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Append one immutable pregame decision snapshot.")
    snapshot.add_argument("--run-id", required=True)
    snapshot.add_argument("--target-date", default="latest")
    snapshot.add_argument("--cutoff-at")
    snapshot.add_argument("--saved-at")
    snapshot.add_argument("--candidates", "--candidate-csv", default=str(DEFAULT_OUTPUT_CSV))
    snapshot.add_argument(
        "--production-predictions",
        "--production-csv",
        dest="production_predictions",
        default=str(DEFAULT_PRODUCTION_PREDICTIONS_CSV),
    )
    snapshot.add_argument(
        "--shadow-predictions",
        "--shadow-csv",
        dest="shadow_predictions",
        default=str(DEFAULT_SHADOW_PREDICTIONS_CSV),
    )
    snapshot.add_argument(
        "--snapshots", "--out", dest="snapshots", default=str(DEFAULT_SNAPSHOTS_CSV)
    )
    snapshot.add_argument("--candidate-pool-source")
    snapshot.add_argument("--lineup-provenance")

    sync = subparsers.add_parser("sync-results", help="Upsert outcomes into the separate result ledger.")
    sync.add_argument("--candidates", "--candidate-csv", default=str(DEFAULT_OUTPUT_CSV))
    sync.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS_CSV))
    sync.add_argument("--results", default=str(DEFAULT_RESULTS_CSV))
    sync.add_argument("--synced-at")

    audit = subparsers.add_parser("audit", help="Audit snapshot immutability and result references.")
    audit.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS_CSV))
    audit.add_argument("--results", default=str(DEFAULT_RESULTS_CSV))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            report = create_snapshot(
                run_id=args.run_id,
                candidates_csv=args.candidates,
                production_predictions_csv=args.production_predictions,
                shadow_predictions_csv=args.shadow_predictions,
                snapshots_csv=args.snapshots,
                target_date=args.target_date,
                cutoff_at=args.cutoff_at,
                saved_at=args.saved_at,
                candidate_pool_source=args.candidate_pool_source,
                lineup_provenance=args.lineup_provenance,
            )
        elif args.command == "sync-results":
            report = sync_results(
                candidates_csv=args.candidates,
                snapshots_csv=args.snapshots,
                results_csv=args.results,
                synced_at=args.synced_at,
            )
        else:
            report = audit_ledger(snapshots_csv=args.snapshots, results_csv=args.results)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("ok", True) else 1
    except (LedgerError, OSError) as exc:
        print(f"prediction-ledger: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
