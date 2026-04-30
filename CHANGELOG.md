# Statbirt Changelog

## 2026-04-29 - Git Baseline

This repository was initialized after the first working version of Statbirt was already built. Earlier project history is summarized here instead of represented as exact chronological commits.

### Built Before Git

- Created the daily MLB hitter candidate pipeline.
- Implemented the v2 weighted model and stop-valve checks.
- Added MLB StatsAPI, Baseball Savant, Open-Meteo, and manual FanGraphs Stuff+ data paths.
- Added postgame result tracking for hit/no-hit outcomes.
- Added bullpen H/IP and relief batting-average support.
- Added weather precipitation probability.
- Built the one-page local web dashboard.
- Added saved dashboard date selection.
- Added row color states for not started, live no hit, hit, final no hit, and postponed.
- Added the Congregation list and dashboard status column.

### Git Tracking Choices

- Source code, tests, docs, and small manual CSV inputs are tracked.
- Large caches, generated daily candidates, and generated dashboard JSON exports are ignored.
- Rebuild generated dashboard data with `python3 -m statbirt.export_web --all-dates --limit 10`.
