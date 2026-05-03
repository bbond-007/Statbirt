const INDEX_URL = "data/dashboard_index.json";
const FALLBACK_DATA_URL = "data/top_picks.json";
let dashboardIndex = null;

const factorOrder = [
  ["hitter", 0],
  ["hitter", 1],
  ["pitching", 1],
  ["pitching", 2],
  ["pitching", 3],
  ["pitching", 4],
  ["matchup", 1],
  ["matchup", 2],
  ["context", 1],
  ["context", 2],
];

function formatDate(value) {
  if (!value) return "Unknown date";
  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) return value;
  return new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(Date.UTC(year, month - 1, day)));
}

function formatShortDate(value) {
  if (!value) return "Unknown date";
  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(Date.UTC(year, month - 1, day)));
}

function isPostponed(pick) {
  return String(pick.game_state || "").toLowerCase() === "postponed";
}

function isHit(pick) {
  return String(pick.game_state || "").toLowerCase() === "hit" || Number(pick.game_hits || 0) > 0;
}

function formatHitRate(picks) {
  const decided = picks.filter((pick) => !isPostponed(pick));
  if (!decided.length) return "--";
  const hits = decided.filter(isHit).length;
  return `${((hits / decided.length) * 100).toFixed(1)}%`;
}

function topTenHitRate(picks) {
  return formatHitRate(picks.filter((pick) => Number(pick.rank || 0) <= 10));
}

function congregationHitRate(picks) {
  return formatHitRate(
    picks.filter((pick) => {
      const status = String(pick.congregation_status || "").trim().toLowerCase();
      return status && status !== "removed";
    })
  );
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function createFactorChip(item) {
  const chip = document.createElement("span");
  chip.className = "factor-chip";

  const label = document.createElement("span");
  label.textContent = item.label;

  const value = document.createElement("strong");
  value.textContent = item.value;

  chip.append(label, value);
  return chip;
}

function formatGameStart(value) {
  const gameDate = value ? new Date(value) : null;
  if (!gameDate || Number.isNaN(gameDate.getTime())) return "TBD";

  const options = {
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  };
  try {
    return new Intl.DateTimeFormat(undefined, options).format(gameDate);
  } catch (error) {
    return new Intl.DateTimeFormat("en-US", {
      ...options,
      timeZone: "America/New_York",
    }).format(gameDate);
  }
}

function formatRainChance(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "N/A";
  return `${parsed.toFixed(1)}%`;
}

function formatTemperature(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "N/A";
  return `${Math.round(parsed)}°F`;
}

function weatherDetail(pick) {
  return `Rain: ${formatRainChance(pick.precip_probability)} · Temp: ${formatTemperature(pick.forecast_temperature_f)}`;
}

function postponedStatus(pick) {
  const text = `${pick.game_state || ""} ${pick.game_status || ""} ${pick.game_state_label || ""}`.toLowerCase();
  return text.includes("postponed");
}

function selectedFactors(pick) {
  const seen = new Set();
  return factorOrder
    .map(([group, index]) => (pick.factors[group] || [])[index])
    .filter((item) => {
      if (!item || item.label === "Rain" || seen.has(item.label)) return false;
      seen.add(item.label);
      return true;
    });
}

function renderPick(pick) {
  const template = document.getElementById("pick-template");
  const row = template.content.firstElementChild.cloneNode(true);
  const stateClass = String(pick.game_state || "unknown").replaceAll("_", "-");
  row.classList.add(`state-${stateClass}`);
  row.title = pick.game_state_label || "";

  row.querySelector(".rank-cell").textContent = pick.rank;
  row.querySelector(".player-name").textContent = pick.player;
  row.querySelector(".matchup-line").textContent = `${pick.team} vs ${pick.opponent} · ${pick.batter_stand} bat vs ${pick.pitcher_hand} arm`;
  row.querySelector(".pitcher-line").textContent = `Probable starter: ${pick.probable_pitcher}`;
  row.querySelector(".venue-line").textContent = `Ballpark: ${pick.venue_name || "TBD"}`;
  row.querySelector(".game-detail-line").textContent =
    `Start: ${formatGameStart(pick.game_start_time_utc)} · ${weatherDetail(pick)}`;
  const statusLine = row.querySelector(".game-status-line");
  if (postponedStatus(pick)) {
    statusLine.textContent = `Game status: ${pick.game_status || "Postponed"}`;
    statusLine.hidden = false;
  } else {
    statusLine.hidden = true;
  }

  const badge = row.querySelector(".pick-badge");
  badge.textContent = pick.pickable ? "Pickable" : "Pass";
  badge.classList.toggle("pickable", pick.pickable);

  const statusCell = row.querySelector(".status-cell");
  const congregationStatus = String(pick.congregation_status || "").trim();
  if (congregationStatus) {
    const statusBadge = document.createElement("span");
    statusBadge.className = `congregation-status status-${congregationStatus.toLowerCase()}`;
    statusBadge.textContent = congregationStatus;
    statusCell.append(statusBadge);
  } else {
    statusCell.textContent = "--";
  }

  row.querySelector(".score-cell strong").textContent = Number(pick.score || 0).toFixed(1);
  row.querySelector(".score-cell span").textContent = "score";

  const factorCell = row.querySelector(".factor-cell");
  selectedFactors(pick).forEach((item) => factorCell.append(createFactorChip(item)));

  const riskCell = row.querySelector(".risk-cell");
  const riskLabel = document.createElement("span");
  riskLabel.className = "risk-label";
  const hardPassReasons = pick.hard_pass_reasons || [];
  const concerns = pick.concerns || [];
  const displayedReasons = hardPassReasons.length ? hardPassReasons : concerns;

  riskLabel.textContent = hardPassReasons.length ? "Stop Valves" : (concerns.length ? "Watch" : "Clean");
  riskCell.append(riskLabel);

  if (displayedReasons.length) {
    const riskList = document.createElement("div");
    riskList.className = "risk-list";
    const tooltip = displayedReasons.join("\n");
    riskList.title = tooltip;
    const item = document.createElement("span");
    item.className = hardPassReasons.length ? "risk-reason hard" : "risk-reason concern";
    item.textContent = displayedReasons[0];
    item.title = tooltip;
    riskList.append(item);
    if (displayedReasons.length > 1) {
      const more = document.createElement("span");
      more.className = "risk-more";
      more.textContent = `+ ${displayedReasons.length - 1} more...`;
      more.title = tooltip;
      riskList.append(more);
    }
    riskCell.append(riskList);
  } else {
    const riskText = document.createElement("span");
    riskText.className = "risk-text";
    riskText.textContent = "No stop valves";
    riskCell.append(riskText);
  }
  return row;
}

function renderSummary(data) {
  const picks = data.picks || [];
  const topScore = picks.length ? Number(picks[0].score || 0).toFixed(1) : "--";
  const pickableArc = data.total_candidates
    ? Math.round((Number(data.pickable_count || 0) / Number(data.total_candidates || 1)) * 360)
    : 0;

  document.documentElement.style.setProperty("--pickable-arc", `${pickableArc}deg`);
  setText("pickable-count", data.pickable_count ?? 0);
  setText("total-candidates", data.total_candidates ?? "--");
  setText("shown-count", picks.length);
  setText("top-score", topScore);
  setText("top-10-hits", topTenHitRate(picks));
  setText("congregation-hits", congregationHitRate(picks));

  const boardDate = formatDate(data.date);
  const modeText = Number(data.pickable_count || 0)
    ? "Showing top scored candidates; pickable rows are labeled."
    : "No candidate cleared every stop-valve; showing top scored candidates for review.";
  const fallbackText = data.used_latest_fallback
    ? ` Requested ${formatDate(data.requested_date)}, so the latest available board is shown.`
    : "";

  setText("board-subtitle", `${boardDate} · ${modeText}${fallbackText}`);
  const pickableText = Number(data.pickable_count || 0) === 1 ? "candidate" : "candidates";
  setText("status-note", data.pickable_count
    ? `${data.pickable_count} ${pickableText} passed all hard stop-valves.`
    : "Strict filters blocked every candidate today.");
}

function renderBoard(data) {
  renderSummary(data);
  const list = document.getElementById("picks");
  list.replaceChildren();

  if (!data.picks || !data.picks.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No candidates available for this date. Run the daily Statbirt export and refresh this page.";
    list.append(empty);
    return;
  }

  data.picks.forEach((pick) => list.append(renderPick(pick)));
}

function dashboardEntryByDate(date) {
  return (dashboardIndex?.dashboards || []).find((entry) => entry.date === date);
}

function dashboardDataUrl(entry) {
  if (!entry?.path) return FALLBACK_DATA_URL;
  return `data/${entry.path}`;
}

function populateDashboardSelect(index) {
  const select = document.getElementById("dashboard-date");
  if (!select) return;
  const dashboards = index?.dashboards || [];
  select.replaceChildren();
  if (!dashboards.length) {
    const option = document.createElement("option");
    option.textContent = "No saved dashboards";
    select.append(option);
    select.disabled = true;
    return;
  }
  dashboards.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.date;
    option.textContent = formatShortDate(entry.date);
    select.append(option);
  });
  select.value = index.active_date || dashboards[0].date;
  select.disabled = dashboards.length < 2;
}

async function loadDashboardDate(date) {
  try {
    const entry = dashboardEntryByDate(date);
    const data = await fetchJson(dashboardDataUrl(entry));
    renderBoard(data);
    const select = document.getElementById("dashboard-date");
    if (select && data.date) select.value = data.date;
  } catch (error) {
    const list = document.getElementById("picks");
    list.innerHTML = `<p class="empty-state">Could not load the selected dashboard. Re-run the web export, then refresh.</p>`;
    setText("board-subtitle", "Data file not loaded.");
    setText("status-note", "Run the web export first.");
  }
}

async function loadBoard() {
  try {
    dashboardIndex = await fetchJson(INDEX_URL);
    populateDashboardSelect(dashboardIndex);
    await loadDashboardDate(dashboardIndex.active_date || dashboardIndex.dashboards?.[0]?.date);
  } catch (error) {
    dashboardIndex = null;
    const select = document.getElementById("dashboard-date");
    if (select) {
      select.replaceChildren();
      const option = document.createElement("option");
      option.textContent = "Current export";
      select.append(option);
      select.disabled = true;
    }
    try {
      const data = await fetchJson(FALLBACK_DATA_URL);
      renderBoard(data);
    } catch (fallbackError) {
      const list = document.getElementById("picks");
      list.innerHTML = `<p class="empty-state">Could not load ${FALLBACK_DATA_URL}. Start the local web server from the Statbirt folder, then refresh.</p>`;
      setText("board-subtitle", "Data file not loaded.");
      setText("status-note", "Run the web export first.");
    }
  }
}

document.getElementById("dashboard-date")?.addEventListener("change", (event) => {
  loadDashboardDate(event.target.value);
});

loadBoard();
