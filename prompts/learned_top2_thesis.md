# Learned Top 2 Thesis Workflow

Use this workflow when the user asks Codex to run today's learned top-2 thesis.

The staging script gathers facts only. The pro thesis, con thesis, and committee decision must be written by ChatGPT/Codex after reviewing the staged data and current public sources.

## Daily Timing

The morning model scripts run at about 7:00am Central and can take around 45 minutes. Stage the context at or after 8:00am Central unless the user explicitly asks otherwise.

## Step 1: Stage Factual Context

Run from the project root:

```powershell
py -3 -m statbirt.stage_learned_thesis --date latest --top 2
```

This writes:

```text
data/manual/learned_top2_context/YYYY-MM-DD.json
```

Read that JSON before writing anything. Treat it as evidence, not as a conclusion.

## Step 2: Review Current Sources

For each player in the staged context, review current free sources when available:

- MLB probable pitchers and live game feed: confirm the starter has not changed.
- MLB starting lineups or a reliable lineup source: confirm batting order and scratch risk.
- MLB injury/news pages: check for injury, rest, or availability notes.
- Baseball Savant: review hitter contact quality, pitcher pitch mix, and matchup shape.
- Bullpen usage source or MLB boxscores: check recent bullpen fatigue and late-game matchup risk.
- Recent game recap/news: look for timing, confidence, role changes, weather, or manager comments.

Use citations or source notes in the final JSON. If a source is unavailable or inconclusive, say that in the note instead of pretending certainty.

## Step 3: Committee Process

Create three analytical roles:

1. Pro advocate: write one dashboard-ready pro thesis paragraph for each player.
2. Con advocate: write one dashboard-ready con thesis paragraph for each player.
3. Reviewer: compare the arguments and choose the single hitter most likely to get a hit.

The reviewer should explain why the chosen player is preferred and why the runner-up was not chosen. The goal is a successful single pick, while acknowledging that the contest allows one or two hitters.

## Step 4: Write Dashboard Thesis JSON

Write the finished ChatGPT/Codex-authored thesis to:

```text
data/manual/learned_top2_theses/YYYY-MM-DD.json
```

Use this schema:

```json
{
  "date": "YYYY-MM-DD",
  "target": "learned_rank_top_2",
  "committee_pick": "Player Name",
  "committee_summary": "One paragraph explaining the single-pick decision.",
  "workflow_notes": [
    "Staged local model evidence first.",
    "Reviewed current free sources before writing the thesis.",
    "Used pro, con, and reviewer committee roles."
  ],
  "source_notes": [
    {
      "label": "Source name",
      "url": "https://example.com",
      "note": "Brief note on what was checked."
    }
  ],
  "players": [
    {
      "rank": 1,
      "player": "Player Name",
      "team": "AAA",
      "opponent": "BBB",
      "probable_pitcher": "Pitcher Name",
      "pro_thesis": "Dashboard-ready paragraph.",
      "con_thesis": "Dashboard-ready paragraph.",
      "committee_thesis": "Dashboard-ready paragraph."
    }
  ]
}
```

## Step 5: Refresh Dashboard

After writing the thesis JSON, run:

```powershell
py -3 -m statbirt.export_learned_web --date latest --limit 5
```

Then verify the learned dashboard at:

```text
http://localhost:8765/learned.html
```

The `Top 2 Committee` section should show each player's Pro Thesis, Con Thesis, and Committee Read, with the committee pick highlighted.
