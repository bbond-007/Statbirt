# Statbirt Project Handoff

Last updated: 2026-05-06

This file exists so Codex on the Desktop PC can pick up the Statbirt project without needing the original chat history.

## Project Roots

- Desktop PC shared/local folder: `compy3600/Coding/Statbirt`
- Mac mount for that same folder: `/Volumes/Coding/Statbirt`
- GitHub repo: `git@github-statbirt:bbond-007/Statbirt.git`
- Current Git branch: `main`

When working from the Desktop PC, use the local Windows path that corresponds to the shared folder. If Python is installed through the Windows launcher, replace `python` with `py -3` in the commands below.

## What Statbirt Does

Statbirt has two parallel approaches for MLB hitter hit-pick ranking:

1. Bob model: the original hand-weighted scoring model with stop valves.
2. Learned model: a data-driven logistic model trained from historical candidate rows and `result_hit` labels.

The daily candidate CSV is the common spine for both models:

```text
data/statbirt_candidates.csv
```

The dashboard reads generated JSON from:

```text
web/data/top_picks.json
web/data/dashboard_index.json
web/data/dashboards/YYYY-MM-DD.json
web/data/learned_shortlist.json
web/data/learned_dashboard_index.json
web/data/learned_dashboards/YYYY-MM-DD.json
```

## Current Data State

As of this handoff:

- `data/statbirt_candidates.csv` has 10,868 rows.
- Candidate dates span `2026-03-25` through `2026-05-06`.
- 43 candidate dates are present.
- 9,742 rows are labeled with `result_hit`.
- `2026-05-06` is the current daily board and is not graded yet.
- 2026 historical backfill through `2026-05-05` is complete.
- The active dashboard is `2026-05-06` with 29 displayed picks.
- `data/manual/stuff_plus.csv` has 688 FanGraphs Stuff+ rows updated `2026-06-16`.

Check current coverage any time:

```bash
python -m statbirt.learned_model audit
```

Verify 2026 historical backfill through May 5 is complete:

```bash
python -m statbirt.backfill --season 2026 --end-date 2026-05-05 --dry-run
```

Expected result at this handoff:

```text
Selected 0 date(s) for backfill.
```

## Learned Model State

Latest local learned-model report:

```text
data/models/hit_probability_report.json
```

Current model at handoff:

- model family: `learned-logistic-v2`
- feature profile: opportunity/contact/pitcher-vulnerability features plus Bob `score`; no stop-valve reason, team, or opponent features
- latest local model version: see `data/models/hit_probability_report.json`
- latest local training rows: 35,363
- latest local training date range: `2025-03-18` through `2026-05-10`
- latest validation AUC: 0.609
- latest validation top-10 hit rate: 75.8%
- walk-forward backtest command: `python scripts/backtest_learned_model_experiments.py --min-train-dates 30`

Score the latest daily candidate rows without retraining:

```bash
python -m statbirt.learned_model score --date latest --top 25
```

Retrain from all labeled rows, then score the latest date:

```bash
python -m statbirt.learned_model run --date latest --top 25
```

Learned-model outputs are local/generated:

```text
data/models/hit_probability_model.json
data/models/hit_probability_report.json
data/model_predictions.csv
```

## Routine Daily Workflow

Use this before games start, after probable pitchers/lineups are reasonable:

```bash
cd path/to/Statbirt
python -m statbirt.cli --date YYYY-MM-DD --top 25 --skip-fangraphs-fetch
python -m statbirt.learned_model score --date latest --top 25
python -m statbirt.export_web --all-dates --limit 10
python -m statbirt.export_learned_web --all-dates --limit 5
```

Why `--skip-fangraphs-fetch`: direct FanGraphs API calls are often Cloudflare-blocked. The manual Stuff+ CSV is the reliable source unless it has just been refreshed.

On the Desktop PC, this pregame sequence is automated by Windows Task Scheduler at 6:30 AM daily. The task is named `Statbirt Daily Morning Run` and runs:

```powershell
X:\Coding\Statbirt\scripts\daily_morning.ps1
```

Logs are written to `logs\daily-morning-*.log`.

After all games are final:

```bash
cd path/to/Statbirt
python -m statbirt.update_results
python -m statbirt.learned_model run --date latest --top 25
python -m statbirt.export_web --all-dates --limit 10
python -m statbirt.export_learned_web --all-dates --limit 5
```

Notes:

- `update_results` fills `result_hit`, `result_hits`, `result_ab`, `result_pa`, `result_status`, and `result_updated_at`.
- `result_status` is machine-readable: `final`, `pending`, `postponed`, `no_appearance`, or `unresolved`.
- Postponed and no-appearance games remain ungraded and should not be counted as misses.
- No-appearance rows keep `result_hit` blank with `result_status=no_appearance`.

## Dashboard Hosting

From the Desktop PC:

```powershell
cd path\to\Statbirt\web
python -m http.server 8765 --bind 0.0.0.0
```

Open on the Desktop PC:

```text
http://localhost:8765
```

From another machine on the network, try:

```text
http://compy3600:8765
```

If the port is stuck on Windows PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen |
  Select-Object -ExpandProperty OwningProcess |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

Then restart the server from `web/`.

## Historical Backfill

The 2026 season through `2026-05-05` is already backfilled in this shared folder.

For 2025 backfill, start with a small chunk:

```bash
python -m statbirt.backfill --season 2025 --start-date 2025-03-18 --end-date 2025-09-28 --max-days 3 --update-results
```

If the chunk looks good, run a longer overnight job. In PowerShell:

```powershell
cd path\to\Statbirt
New-Item -ItemType Directory -Force logs | Out-Null
python -m statbirt.backfill --season 2025 --start-date 2025-03-18 --end-date 2025-09-28 --update-results --train-learned-model 2>&1 |
  Tee-Object -FilePath ("logs\backfill-2025-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
```

Backfill is slow because it builds Baseball Savant/Statcast feature stores. That is normal. Do not delete `data/cache/` unless you are comfortable rebuilding those caches.

For long 2025 work, prefer explicit `--start-date` and `--end-date` values. MLB's schedule API can include rescheduled/makeup metadata, and explicit windows keep the run predictable.

## Important Source Notes

- MLB StatsAPI supplies schedules, probable pitchers, teams, boxscores, and results.
- Baseball Savant/pybaseball supplies pitch-level H2H, split, whiff/K, xBA, pitch-type, sprint speed, and park context. It is the slowest source.
- Open-Meteo supplies rain probability and first-pitch temperature.
- FanGraphs Stuff+ is maintained in `data/manual/stuff_plus.csv`; direct API fetches commonly fail with 403.
- `data/manual/congregation.csv` controls the dashboard Congregation status column.

## Git And Generated Data

Track code, docs, tests, and manual inputs. Do not commit generated data or caches:

```text
data/statbirt_candidates.csv
data/model_predictions.csv
data/models/
data/cache/
web/data/*.json
logs/
```

Useful checks:

```bash
git status --short --branch
python -m pytest -q
python -m statbirt.learned_model audit
```

The shared SMB repo is configured with:

```bash
git config core.filemode false
```

That prevents Windows/share executable-bit changes from appearing as false Git diffs.

## Key Files

- `statbirt/cli.py`: daily Bob model entry point
- `statbirt/pipeline.py`: candidate assembly
- `statbirt/scoring.py`: Bob model weights and stop valves
- `statbirt/results.py`: postgame result updater
- `statbirt/export_web.py`: dashboard JSON exporter
- `statbirt/export_learned_web.py`: learned dashboard JSON exporter
- `statbirt/learned_model.py`: learned model train/score/audit
- `statbirt/backfill.py`: historical backfill helper
- `web/`: dashboard frontend
- `data/manual/stuff_plus.csv`: manual Stuff+ source
- `data/manual/congregation.csv`: curated Congregation list
