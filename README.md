# Statbirt

Statbirt ranks MLB hitters by how attractive they are for a same-day "to get a hit" pick.

This is a fresh project inspired by `/Users/blake/Coding/Baseball`, but the scoring model is intentionally rebuilt around the new weight buckets and hard stop-valves.

## Current Status

As of April 30, 2026:

- Work is continuing in `/Users/blake/Coding/Statbirt`; no separate `Statbirt_v2` rebuild has been needed so far.
- `data/manual/stuff_plus.csv` contains 526 normalized 2026 FanGraphs Stuff+ rows pulled through the FanGraphs leaderboard/API browser fallback.
- `data/statbirt_candidates.csv` has been regenerated with current Stuff+ values, so `pitcher_stuff_plus` is populated for the current candidate rows.
- `data/statbirt_candidates.csv` has Column R / `precip_probability` populated for the current candidate rows via the weather-only updater.
- A full production-style run should avoid `--skip-savant`; that flag is only for faster smoke checks because it leaves Savant-dependent discipline and split columns blank.

## Daily Run

```bash
cd /Users/blake/Coding/Statbirt
python3 -m statbirt.cli --date YYYY-MM-DD --top 25
```

The default output is:

```text
/Users/blake/Coding/Statbirt/data/statbirt_candidates.csv
```

Daily runs upsert into the CSV instead of wiping it. Ungraded rows for the same date are replaced by the newest model output, while rows that already have postgame result columns filled in are preserved.

Use a concrete game date when backfilling. For example:

```bash
python3 -m statbirt.cli --date 2026-04-26 --top 25
```

## Postgame Results

After games are final, update every candidate row with boxscore results:

```bash
cd /Users/blake/Coding/Statbirt
python3 -m statbirt.update_results
```

That fills:

- `result_hit`: `1` if the hitter had at least one hit, `0` if he appeared and went hitless, blank for no batting appearance
- `result_hits`
- `result_ab`
- `result_pa`
- `result_status`
- `result_updated_at`
- `notes`

Use `--refresh-filled` to recalculate rows that already have results, or `--dry-run` to see how many rows would change.

## Weather Backfill

If Column R / `precip_probability` is blank but the rest of the candidate CSV looks good, refresh only the weather column:

```bash
cd /Users/blake/Coding/Statbirt
python3 -m statbirt.update_weather
```

Use `--dry-run` first to preview the row count, or `--refresh-filled` to recalculate weather values that are already present.

## Bullpen Backfill

If `bullpen_opp_ba` / `Relief BA` is missing but the rest of the candidate CSV looks good, refresh only the bullpen relief columns:

```bash
cd /Users/blake/Coding/Statbirt
python3 -m statbirt.update_bullpen
```

Use `--dry-run` first to preview the row count, or `--refresh-filled` to recalculate values that are already present. New daily model runs write `bullpen_opp_ba` automatically.
Bullpen values are built from MLB boxscores through the day before the candidate date and skip floating-time doubleheader Game 2 placeholders that MLB excludes from its official team reliever split tables.

## Web Dashboard

The one-page dashboard lives in:

```text
/Users/blake/Coding/Statbirt/web/index.html
```

Refresh its data after the daily CSV is current:

```bash
cd /Users/blake/Coding/Statbirt
python3 -m statbirt.export_web --date YYYY-MM-DD --limit 10
```

To rebuild the active dashboard and every saved date in the CSV:

```bash
python3 -m statbirt.export_web --all-dates --limit 10
```

Then serve the page locally:

```bash
cd /Users/blake/Coding/Statbirt/web
python3 -m http.server 8765
```

Open `http://localhost:8765`. The dashboard shows the top scored candidates for the selected date and labels rows that clear every stop-valve as `Pickable`; blocked rows show `PASS`. The date dropdown is populated from `web/data/dashboard_index.json` and loads saved dashboards from `web/data/dashboards/YYYY-MM-DD.json`.

The Bullpen score bucket still follows the v2 instruction sheet and uses opposing bullpen H/IP. The dashboard also shows `Relief BA`, calculated from MLB boxscore relief pitching hits allowed divided by relief pitching at-bats allowed, as a companion caution signal.

Each player row includes the probable starter, ballpark, first-pitch time formatted by the browser's timezone, and the rain chance. The compact Key Factors chips intentionally omit Rain because it is already shown in the player details.

The dashboard also reads the Congregation list at:

```text
/Users/blake/Coding/Statbirt/data/manual/congregation.csv
```

The normal top 10 by score are always shown, then any Congregation players from that date's candidate CSV are added even if their model rank is outside the top 10. The dashboard `Status` column comes from this file, currently using `Publisher` and `Removed`.

The summary cards show `Top 10 Hits` and `Congregation Hits` as hit percentages for completed/non-postponed games. Postponed games are excluded from those denominators.

Dashboard row colors come from the MLB schedule/boxscore snapshot taken during `statbirt.export_web`:

- no color change: game has not started
- light gray: game is postponed
- light yellow: game has started and the hitter has 0 hits
- light green: hitter has at least 1 hit
- light red: game is final and the hitter finished with 0 hits

If MLB marks a game postponed, the Player column shows `Game status: Postponed` so it is not mistaken for a played final.

Useful flags:

```bash
python3 -m statbirt.cli --skip-savant
python3 -m statbirt.cli --skip-weather
python3 -m statbirt.cli --skip-fangraphs-fetch
python3 -m statbirt.cli --skip-bullpen
python3 -m statbirt.cli --hitter-play-log-seasons 7
python3 -m statbirt.cli --strict-missing-stop-data
python3 -m statbirt.cli --stuff-plus-csv data/manual/stuff_plus.csv
python3 -m statbirt.export_web --congregation-csv data/manual/congregation.csv
```

`--skip-savant`, `--skip-weather`, `--skip-fangraphs-fetch`, and `--skip-bullpen` are useful for debugging source-specific failures. They should not be used for a final daily board unless the missing source is intentionally being ignored.

## Model Shape

The score is a 0-100 weighted score:

- Hitter: 25%
  - 10% HiPA over last/up-to 2500 PA
  - 7% PA per game this season
  - 5% HiPA over last/up-to 500 PA
  - 3% HiPA over the PA span needed to reach last/up-to 75 AB
- Starting pitcher: 25%
- H2H: 20%
- Bullpen/team pitching: 15%
- Other context: 15%

Stop-valves are evaluated separately. A hitter can have a strong score and still be marked `pickable=N` if a hard-pass condition is triggered.

The v2 instruction sheet lists four 3% items under Other while keeping the bucket total at 15%. Statbirt keeps the original 3% lineup-opportunity item as the implicit fifth Other input until that missing 3% is renamed.

## Stop-Valves

Current hard-pass checks include:

- minimum 2 PA against the probable starter
- doubleheader
- hitter below .270 in both long and last-500 AB windows
- precipitation probability above 40%
- opener risk
- hitter season or last-500 PA BB rate above 12%
- hitter season PA/G below 4.2
- hitter season or last-500 PA whiff rate above 25%
- hitter season or last-500 PA K rate above 22%
- H2H whiff rate above 25% or H2H K rate above 20%
- hitter below .150 over last 25 AB
- starter under .875 H/IP in last 200 IP or season
- starter under 10 hits in last 18 IP
- starter Stuff+ above 95
- hitter L/R split requirements
- starter same-handed opponent BA above .245 over both 50 and 200 IP windows

## Source Notes

Most features come from MLB StatsAPI and Baseball Savant:

- rolling hitter PA/AB windows
- hitter PA/G from season boxscore usage
- hitter BB, whiff, and K rates from Savant pitch-level data
- starter recent hits per inning
- current-season starter hits per inning
- H2H plate appearances, whiff rate, K rate, exit velocity, and xBA
- hitter rolling L/R split estimates
- inferred pitch-type matchup
- bullpen hits per inning from boxscores
- bullpen opponent batting average from relief pitcher hits allowed divided by relief pitcher at-bats allowed
- sprint speed and park hit factors

Direct H2H uses career Statcast matchup rows from March 1, 2015 through the day before the candidate date, matching Baseball Savant's Player Matchup table. The broader Savant store for hitter discipline, hitter splits, inferred pitch-type matchup, and pitcher L/R split context still uses the configurable recent-season window controlled by `--savant-years`.

Weather uses MLB venue coordinates plus Open-Meteo:

- The schedule payload gives each game venue and first pitch time.
- When the schedule payload does not include coordinates, Statbirt hydrates the MLB venue endpoint with `location,timezone`.
- `statbirt/weather.py` asks Open-Meteo for hourly `precipitation_probability` at the venue coordinates and uses the maximum value during the four-hour game window after first pitch.
- A small venue-coordinate override exists for Estadio Alfredo Harp Helu because MLB returns city/country for that Mexico City venue but not exact coordinates.

FanGraphs Stuff+ is isolated in `statbirt/fangraphs.py`. The public FanGraphs leaderboard exposes Stuff+ under Major League Pitching, `type=36`, but direct local API requests can be Cloudflare-blocked. For reliability, Statbirt supports a manual file at:

```text
/Users/blake/Coding/Statbirt/data/manual/stuff_plus.csv
```

The manual file is now the preferred reliable path for Stuff+. The daily model reads it automatically, and `--skip-fangraphs-fetch` is safe when the manual file is current.

If FanGraphs export access is available:

1. Open the FanGraphs Major League pitching leaderboard.
2. Set the season to the current season.
3. Choose the Pitch Modeling / Stuff+ view.
4. Set the playing-time qualifier low enough to include probable starters.
5. Click `Export` on the leaderboard.
6. Import the downloaded CSV:

```bash
cd /Users/blake/Coding/Statbirt
python3 -m statbirt.import_stuff_plus ~/Downloads/FanGraphs\ Leaderboard.csv --season 2026
```

The importer writes the normalized file to `data/manual/stuff_plus.csv`, which the daily model reads automatically.

On April 27, 2026, the FanGraphs leaderboard's visible `Data Export` button appeared as `Members Only`. The browser/API fallback still worked from the loaded leaderboard page, and the API field used for Stuff+ was `sp_stuff`. If the official export is blocked again, use the browser leaderboard/API route or recreate the normalized manual CSV with these columns:

```text
season,player_id,player,team,stuff_plus,source,updated
```

Leaderboard route used for the 2026 pull:

```text
https://www.fangraphs.com/leaders/major-league?stats=pit&type=36&season=2026&qual=0
```

## Candidate Columns

Some columns are intentionally blank until their source data is loaded:

- Columns AE-AN depend on Baseball Savant pitch-level data. They are blank when the daily run uses `--skip-savant`.
- Column AS is `pitcher_stuff_plus`. It depends on FanGraphs Stuff+, preferably from `data/manual/stuff_plus.csv`.
- Column R is `precip_probability`. It depends on weather being enabled and on usable venue coordinates.
- `game_start_time_utc` and `venue_name` are written by new daily runs from the MLB schedule payload and are also backfilled into web dashboard exports when older CSV rows do not have them.
- Result columns stay blank until `python3 -m statbirt.update_results` is run after games are final.

The Savant-dependent AE-AN columns are:

```text
hitter_bb_rate_season
hitter_bb_rate_500_pa
hitter_whiff_rate_season
hitter_whiff_rate_500_pa
hitter_k_rate_season
hitter_k_rate_500_pa
hitter_split_ba_500_vs_lhp
hitter_split_ba_500_vs_rhp
hitter_split_ba_1500_vs_lhp
hitter_split_ba_1500_vs_rhp
```

## Files

- `statbirt/scoring.py`: weights and stop-valve evaluation
- `statbirt/pipeline.py`: daily candidate assembly and CSV export
- `statbirt/mlb_api.py`: MLB StatsAPI client and rolling usage/game-log helpers
- `statbirt/savant.py`: Baseball Savant pitch-level matchup/split helpers
- `statbirt/fangraphs.py`: Stuff+ fetch/manual lookup
- `statbirt/import_stuff_plus.py`: normalize a downloaded FanGraphs Stuff+ CSV
- `statbirt/update_results.py`: postgame candidate result updater
- `statbirt/update_weather.py`: weather-only precipitation probability updater
- `statbirt/update_bullpen.py`: bullpen relief BA/H-IP updater
- `statbirt/export_web.py`: export top-pick JSON for the web dashboard
- `statbirt/weather.py`: precipitation lookup via Open-Meteo
- `data/manual/congregation.csv`: optional friend-curated player list and dashboard status labels
- `scripts/run_daily.py`: convenience wrapper for the daily model
- `scripts/import_stuff_plus.py`: convenience wrapper for Stuff+ import
- `scripts/update_results.py`: convenience wrapper for postgame result updates
- `scripts/update_weather.py`: convenience wrapper for weather backfill
- `scripts/update_bullpen.py`: convenience wrapper for bullpen relief backfill
- `scripts/export_web.py`: convenience wrapper for web dashboard export
- `web/index.html`: one-page Statbirt top-picks dashboard
- `tests/test_scoring.py`: scoring and valve unit tests

## Data Caveats

The model treats missing stop-valve data as a concern by default, except for direct H2H PA, which is a hard requirement. Use `--strict-missing-stop-data` if you want missing stop-valve fields to become hard passes.

The "Other" 3% slot is implemented as lineup opportunity, using expected plate appearances from batting-order slot.

## GitHub Workflow

This repo is meant to track the durable project, not every generated data artifact.

GitHub repo:

```text
https://github.com/bbond-007/Statbirt
```

The local `origin` remote uses a repo-specific SSH deploy key:

```text
git@github-statbirt:bbond-007/Statbirt.git
```

The SSH alias is configured in `~/.ssh/config` and points at the private key `~/.ssh/statbirt_github_ed25519`. The matching public key is registered on GitHub as the read/write deploy key named `Statbirt Mac deploy key`. Do not commit SSH keys, tokens, downloaded exports, or generated dashboard data.

Tracked:

- source code in `statbirt/`, `scripts/`, `web/`, and `tests/`
- documentation
- requirements
- small manual inputs in `data/manual/`

Ignored:

- `data/cache/`
- `data/statbirt_candidates.csv`
- generated dashboard JSON in `web/data/`
- Python caches and local environment files

Daily generated files still live on this machine and are rebuilt by the scripts. If a fresh clone needs the dashboard data, first run the daily model or copy in a candidate CSV, then run:

```bash
python3 -m statbirt.export_web --all-dates --limit 10
```

Going forward, make a small commit after each meaningful project change. The commit history becomes the project memory that a future Codex session can inspect even if chat context is gone.

Typical change flow:

```bash
git status
git add README.md statbirt tests scripts web data/manual
git commit -m "Describe the change"
git push
```

When verifying code changes, `python3 -m compileall -q statbirt scripts` catches syntax/import issues. After installing the dependencies from `requirements.txt`, run the test suite with:

```bash
python3 -m pytest -q
```

If `pytest` is not installed in the active Python environment, use `python3 -m compileall -q statbirt scripts` as the quick built-in smoke check until the environment is set up.
