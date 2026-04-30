from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from .config import DEFAULT_OUTPUT_CSV
from .mlb_api import MLBClient, batting_plate_appearances
from .utils import TEAM_ID_BY_ABBR, canonical_date, normalize_name, parse_int

RESULT_FIELDS = [
    "result_hit",
    "result_hits",
    "result_ab",
    "result_pa",
    "result_status",
    "result_updated_at",
    "notes",
]


def parse_results_date(value: str) -> date | None:
    return canonical_date(value)


def team_id_from_value(value) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = parse_int(text)
    if parsed is not None:
        return parsed
    return TEAM_ID_BY_ABBR.get(text.upper())


def append_note(row: dict[str, str], note: str) -> bool:
    note = (note or "").strip()
    if not note:
        return False
    existing = (row.get("notes") or "").strip()
    if not existing:
        row["notes"] = note
        return True
    parts = [part.strip() for part in existing.split(" | ") if part.strip()]
    if note not in parts:
        parts.append(note)
        row["notes"] = " | ".join(parts)
        return True
    return False


def ensure_fieldnames(fieldnames) -> list[str]:
    ordered = []
    for field in fieldnames or []:
        if field and field not in ordered:
            ordered.append(field)
    for field in RESULT_FIELDS:
        if field not in ordered:
            ordered.append(field)
    return ordered


def coerce_row(row: dict[str, str], fieldnames: list[str]) -> dict[str, str]:
    return {field: "" if row.get(field) is None else str(row.get(field, "")) for field in fieldnames}


def load_candidates_table(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return [], list(RESULT_FIELDS)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = ensure_fieldnames(reader.fieldnames or [])
        rows = [coerce_row(row, fieldnames) for row in reader]
    return rows, fieldnames


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def key(row: dict[str, str]):
        row_date = parse_results_date(row.get("date"))
        ordinal = row_date.toordinal() if row_date else 0
        score = row.get("score")
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = -1.0
        return (-ordinal, -score_value, normalize_name(row.get("player", "")))

    return sorted(rows, key=key)


def write_candidates_table(path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = ensure_fieldnames(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(sort_rows([coerce_row(row, output_fields) for row in rows]))


def row_key(row: dict[str, str]):
    row_date = (parse_results_date(row.get("date")) or row.get("date") or "")
    row_date = row_date.isoformat() if isinstance(row_date, date) else str(row_date)
    player_id = str(row.get("player_id", "")).strip()
    game_pk = str(row.get("game_pk", "")).strip()
    if row_date and player_id and game_pk:
        return ("game", row_date, player_id, game_pk)
    return (
        "fallback",
        row_date,
        player_id,
        normalize_name(row.get("player", "")),
        str(row.get("team", "")).strip(),
        str(row.get("opponent", "")).strip(),
    )


def has_result_data(row: dict[str, str]) -> bool:
    return any(str(row.get(field, "")).strip() for field in RESULT_FIELDS if field != "notes")


def merge_candidate_row(existing: dict[str, str], incoming: dict[str, str], fieldnames: list[str]) -> dict[str, str]:
    merged = coerce_row(existing, fieldnames)
    for field in fieldnames:
        incoming_value = incoming.get(field, "")
        if field in RESULT_FIELDS and str(merged.get(field, "")).strip() and not str(incoming_value).strip():
            continue
        if incoming_value not in ("", None):
            merged[field] = str(incoming_value)
    return merged


def upsert_candidate_rows(
    path: str | Path,
    new_rows: list[dict[str, str]],
    *,
    replace_date: str | None = None,
) -> dict[str, int]:
    rows, fieldnames = load_candidates_table(path)
    for row in new_rows:
        for field in row:
            if field and field not in fieldnames:
                fieldnames.append(field)
    fieldnames = ensure_fieldnames(fieldnames)
    if replace_date:
        rows = [
            row for row in rows
            if (parse_results_date(row.get("date")) or row.get("date")) != parse_results_date(replace_date)
            or has_result_data(row)
        ]
    indexes = {row_key(row): idx for idx, row in enumerate(rows)}
    inserted = 0
    updated = 0
    for raw_row in new_rows:
        incoming = coerce_row(raw_row, fieldnames)
        key = row_key(incoming)
        idx = indexes.get(key)
        if idx is None:
            rows.append(incoming)
            indexes[key] = len(rows) - 1
            inserted += 1
        else:
            rows[idx] = merge_candidate_row(rows[idx], incoming, fieldnames)
            updated += 1
    write_candidates_table(path, rows, fieldnames)
    return {"inserted": inserted, "updated": updated, "total_rows": len(rows)}


def is_final_status(status_text: str) -> bool:
    status = (status_text or "").strip().lower()
    return any(token in status for token in ("final", "game over", "completed"))


def _game_status(game: dict) -> str:
    status = game.get("status") or {}
    if isinstance(status, dict):
        return status.get("detailedState") or status.get("abstractGameState") or ""
    return str(status or "")


def _load_day_games(client: MLBClient, day: date) -> list[dict]:
    output = []
    for game in client.schedule(day, day, hydrate=""):
        game_pk = parse_int(game.get("gamePk"))
        teams = game.get("teams") or {}
        away_id = parse_int(((teams.get("away") or {}).get("team") or {}).get("id"))
        home_id = parse_int(((teams.get("home") or {}).get("team") or {}).get("id"))
        status = _game_status(game)
        game_record = {
            "game_pk": game_pk,
            "away_id": away_id,
            "home_id": home_id,
            "status": status,
            "players_by_id": {},
            "players_by_name": {},
        }
        if game_pk is None or not is_final_status(status):
            output.append(game_record)
            continue
        try:
            box = client.boxscore(game_pk)
        except Exception:
            output.append(game_record)
            continue
        for side in ("away", "home"):
            side_box = ((box.get("teams") or {}).get(side) or {})
            team_id = parse_int(((side_box.get("team") or {}).get("id")))
            batting_order = {parse_int(pid) for pid in (side_box.get("batters") or [])}
            for key, player_block in (side_box.get("players") or {}).items():
                player_id = parse_int(str(key).replace("ID", ""))
                if player_id is None:
                    continue
                person = player_block.get("person") or {}
                full_name = person.get("fullName") or ""
                batting = ((player_block.get("stats") or {}).get("batting") or {})
                hits = parse_int(batting.get("hits")) or 0
                at_bats = parse_int(batting.get("atBats")) or 0
                plate_appearances = batting_plate_appearances(batting)
                appeared = player_id in batting_order or at_bats > 0 or plate_appearances > 0
                result = {
                    "player_id": player_id,
                    "name": full_name,
                    "team_id": team_id,
                    "hits": hits,
                    "at_bats": at_bats,
                    "plate_appearances": plate_appearances,
                    "appeared": appeared,
                }
                game_record["players_by_id"][player_id] = result
                if full_name:
                    game_record["players_by_name"].setdefault(normalize_name(full_name), []).append(result)
        output.append(game_record)
    return output


def _candidate_games_for_row(row: dict[str, str], games: list[dict]) -> list[dict]:
    game_pk = parse_int(row.get("game_pk"))
    if game_pk is not None:
        return [game for game in games if game.get("game_pk") == game_pk]
    team_id = team_id_from_value(row.get("team_id") or row.get("team"))
    opponent_id = team_id_from_value(row.get("opponent_id") or row.get("opponent"))
    filtered = games
    if team_id is not None:
        filtered = [game for game in filtered if team_id in {game.get("away_id"), game.get("home_id")}]
    if opponent_id is not None:
        filtered = [game for game in filtered if opponent_id in {game.get("away_id"), game.get("home_id")}]
    return filtered


def _resolve_row_result(row: dict[str, str], final_games: list[dict]):
    player_id = parse_int(row.get("player_id"))
    if player_id is not None:
        matches = [
            (game, game["players_by_id"][player_id])
            for game in final_games
            if player_id in game.get("players_by_id", {})
        ]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, "Multiple final games matched this player_id."

    player_name = normalize_name(row.get("player"))
    matches = []
    for game in final_games:
        matches.extend((game, result) for result in game.get("players_by_name", {}).get(player_name, []))
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, "Multiple final games matched this player name."
    return None, "No final-game player match found."


def _mark_no_appearance(row: dict[str, str], game: dict, note: str) -> None:
    row["game_pk"] = str(game.get("game_pk") or row.get("game_pk") or "")
    row["result_hit"] = ""
    row["result_hits"] = "0"
    row["result_ab"] = "0"
    row["result_pa"] = "0"
    row["result_status"] = f"{game.get('status') or 'Final'} - no appearance"
    row["result_updated_at"] = datetime.now().isoformat(timespec="seconds")
    append_note(row, note)


def update_results_csv(
    path: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    refresh_filled: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    rows, fieldnames = load_candidates_table(path)
    if not rows:
        return {"rows": 0, "updated": 0, "pending": 0}
    client = MLBClient()
    today = date.today()
    rows_by_date: dict[date, list[tuple[int, dict[str, str]]]] = {}
    for idx, row in enumerate(rows):
        if not refresh_filled and has_result_data(row):
            continue
        row_date = parse_results_date(row.get("date"))
        if row_date is None or row_date > today:
            continue
        rows_by_date.setdefault(row_date, []).append((idx, row))

    updated = 0
    pending = 0
    for row_date, indexed_rows in sorted(rows_by_date.items()):
        games = _load_day_games(client, row_date)
        if not games:
            continue
        for idx, row in indexed_rows:
            before = dict(row)
            possible_games = _candidate_games_for_row(row, games)
            if not possible_games:
                append_note(row, "No matching scheduled game found.")
            else:
                final_games = [game for game in possible_games if is_final_status(game.get("status", ""))]
                if not final_games:
                    row["result_status"] = possible_games[0].get("status", "") or row.get("result_status", "")
                    pending += 1
                else:
                    resolved, error = _resolve_row_result(row, final_games)
                    if resolved is None:
                        if len(final_games) == 1:
                            _mark_no_appearance(row, final_games[0], error or "No batting appearance found.")
                        else:
                            append_note(row, error or "Unable to match final game result.")
                    else:
                        game, result = resolved
                        hits = int(result.get("hits") or 0)
                        row["game_pk"] = str(game.get("game_pk") or row.get("game_pk") or "")
                        row["result_hits"] = str(hits)
                        row["result_ab"] = str(result.get("at_bats") or 0)
                        row["result_pa"] = str(result.get("plate_appearances") or 0)
                        row["result_status"] = game.get("status") or "Final"
                        row["result_updated_at"] = datetime.now().isoformat(timespec="seconds")
                        row["result_hit"] = "1" if hits > 0 else "0"
                        if not result.get("appeared"):
                            _mark_no_appearance(row, game, "No batting appearance found in final boxscore.")
            if row != before:
                rows[idx] = row
                updated += 1
    if updated and not dry_run:
        write_candidates_table(path, rows, fieldnames)
    return {"rows": len(rows), "updated": updated, "pending": pending}
