# Statbirt Agent Notes

Start here when working on this project from Codex, especially on the Desktop PC.

1. Read `PROJECT_HANDOFF.md` first, then `README.md`.
2. Use the project root as the working directory.
   - Desktop PC local/shared folder: `compy3600/Coding/Statbirt` or the local Windows path that points to it.
   - Mac mount path for the same share: `/Volumes/Coding/Statbirt`.
3. Generated data is intentionally local and ignored by Git:
   - `data/statbirt_candidates.csv`
   - `data/model_predictions.csv`
   - `data/models/`
   - `data/cache/`
   - `web/data/*.json`
   - `logs/`
4. Before changing code, run `git status --short --branch`.
5. Before finishing code changes, run `python -m pytest -q` from the project root.
6. The shared SMB copy is configured with `core.filemode=false`; ignore executable-bit noise if another clone shows it.
7. Do not delete `data/cache/` casually. Backfills and daily runs use it heavily, especially Baseball Savant/Statcast caches.
