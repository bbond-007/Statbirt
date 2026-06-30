const INDEX_URL = "data/learned_dashboard_index.json";
const FALLBACK_DATA_URL = "data/learned_shortlist.json";
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

const roofedBallparks = new Set([
  "american family field",
  "miller park",
  "chase field",
  "daikin park",
  "minute maid park",
  "globe life field",
  "loandepot park",
  "marlins park",
  "rogers centre",
  "skydome",
  "t mobile park",
  "safeco field",
  "tropicana field",
]);

function normalizeVenueName(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function isRoofedBallpark(pick) {
  if (pick.roofed_ballpark === true) return true;
  return roofedBallparks.has(normalizeVenueName(pick.venue_name));
}

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

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function percent(value, digits = 1) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "--";
  return `${(parsed * 100).toFixed(digits)}%`;
}

function numberText(value, digits = 1) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "--";
  return parsed.toFixed(digits);
}

function formatRainChance(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "N/A";
  return `${parsed.toFixed(1)}%`;
}

function formatTemperature(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "N/A";
  return `${Math.round(parsed)}\u00B0F`;
}

function weatherDetail(pick) {
  const weatherLabel = pick.weather_label || (isRoofedBallpark(pick) ? "Dome" : `Rain: ${formatRainChance(pick.precip_probability)}`);
  return `${weatherLabel} \u00B7 Temp: ${formatTemperature(pick.forecast_temperature_f)}`;
}

function formatBattingAverage(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "--";
  return parsed.toFixed(3).replace(/^0/, "");
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

function isHit(pick) {
  const gameState = String(pick.game_state || "").toLowerCase();
  return gameState === "hit" || pick.result_hit === true || Number(pick.result_hits || pick.game_hits || 0) > 0;
}

function finalGameState(pick) {
  const gameStatus = `${pick.game_status || ""} ${pick.game_state_label || ""}`.toLowerCase();
  return gameStatus.includes("final") || gameStatus.includes("game over") || gameStatus.includes("completed");
}

function effectiveResultStatus(pick) {
  const status = String(pick.result_status || "").toLowerCase();
  const gameState = String(pick.game_state || "").toLowerCase();
  if (status) return status;
  if (gameState === "postponed") return "postponed";
  if (gameState === "final_no_hit") return "final";
  if (gameState === "hit" && finalGameState(pick)) return "final";
  if (gameState === "unknown") return "pending";
  return status || "pending";
}

function resultLabel(pick) {
  const status = effectiveResultStatus(pick);
  if (status === "final") return isHit(pick) ? "Hit" : "No hit";
  if (status === "postponed") return "Postponed";
  if (status === "no_appearance") return "No PA";
  if (status === "unresolved") return "Unresolved";
  return "Pending";
}

function resultClass(pick) {
  const gameState = String(pick.game_state || "").toLowerCase();
  if (gameState === "hit") return "hit";
  if (gameState === "final_no_hit") return "miss";
  if (gameState === "postponed") return "postponed";
  const status = effectiveResultStatus(pick);
  if (status === "final") return isHit(pick) ? "hit" : "miss";
  if (status === "postponed") return "postponed";
  if (status === "no_appearance") return "no-appearance";
  if (status === "unresolved") return "unresolved";
  return "pending";
}

function resultDetail(pick) {
  const status = effectiveResultStatus(pick);
  if (status === "postponed" || status === "pending" || status === "unresolved") return "";
  if (status === "no_appearance") return "0 PA";
  const hits = pick.result_hits ?? pick.game_hits ?? 0;
  const atBats = pick.result_ab ?? 0;
  const plateAppearances = pick.result_pa ?? 0;
  return `${hits}-${atBats}, ${plateAppearances} PA`;
}

function topFiveResult(picks) {
  const decided = picks.filter((pick) => effectiveResultStatus(pick) === "final");
  if (!decided.length) return "--";
  const hits = decided.filter(isHit).length;
  return `${hits}/${decided.length}`;
}

function createChip(text, className = "") {
  const chip = document.createElement("span");
  chip.className = `factor-chip learned-chip ${className}`.trim();
  const label = document.createElement("span");
  label.textContent = text;
  chip.append(label);
  return chip;
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

function selectedFactors(pick) {
  const seen = new Set();
  return factorOrder
    .map(([group, index]) => (pick.factors?.[group] || [])[index])
    .filter((item) => {
      if (!item || item.label === "Rain" || seen.has(item.label)) return false;
      seen.add(item.label);
      return true;
    });
}

let stopValveReturnFocus = null;

function closeStopValveDialog() {
  const dialog = document.getElementById("stop-valve-dialog");
  if (!dialog || dialog.hidden) return;
  dialog.hidden = true;
  document.body.classList.remove("dialog-open");
  if (stopValveReturnFocus) {
    stopValveReturnFocus.focus();
    stopValveReturnFocus = null;
  }
}

function ensureStopValveDialog() {
  let dialog = document.getElementById("stop-valve-dialog");
  if (dialog) return dialog;

  dialog = document.createElement("div");
  dialog.id = "stop-valve-dialog";
  dialog.className = "stop-dialog";
  dialog.hidden = true;
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  dialog.setAttribute("aria-labelledby", "stop-dialog-title");

  const panel = document.createElement("section");
  panel.className = "stop-dialog-panel";

  const header = document.createElement("div");
  header.className = "stop-dialog-header";

  const titleWrap = document.createElement("div");
  const title = document.createElement("h2");
  title.id = "stop-dialog-title";
  const subtitle = document.createElement("p");
  subtitle.className = "stop-dialog-subtitle";
  titleWrap.append(title, subtitle);

  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "stop-dialog-close";
  closeButton.textContent = "\u00D7";
  closeButton.setAttribute("aria-label", "Close stop valves");
  closeButton.addEventListener("click", closeStopValveDialog);

  const list = document.createElement("ul");
  list.className = "stop-dialog-list";

  header.append(titleWrap, closeButton);
  panel.append(header, list);
  dialog.append(panel);
  document.body.append(dialog);

  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) closeStopValveDialog();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeStopValveDialog();
  });

  return dialog;
}

function openStopValveDialog({ player, reasons, hasHardPass, trigger }) {
  const dialog = ensureStopValveDialog();
  stopValveReturnFocus = trigger || document.activeElement;

  const title = dialog.querySelector("#stop-dialog-title");
  const subtitle = dialog.querySelector(".stop-dialog-subtitle");
  const list = dialog.querySelector(".stop-dialog-list");
  const closeButton = dialog.querySelector(".stop-dialog-close");
  const label = hasHardPass ? "Stop Valves" : "Watch Notes";

  title.textContent = `${label}: ${player || "Candidate"}`;
  subtitle.textContent = `${reasons.length} ${reasons.length === 1 ? "item" : "items"}`;
  list.replaceChildren();

  reasons.forEach((reason) => {
    const item = document.createElement("li");
    item.className = hasHardPass ? "hard" : "concern";
    item.textContent = reason;
    list.append(item);
  });

  dialog.hidden = false;
  document.body.classList.add("dialog-open");
  closeButton.focus();
}

function appendStopValveDetailsButton(container, pick, reasons, hasHardPass, tooltip) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "risk-detail-button";
  button.textContent = "View all";
  button.title = tooltip;
  button.setAttribute(
    "aria-label",
    `Show all ${hasHardPass ? "stop valves" : "watch notes"} for ${pick.player || "candidate"}`
  );
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    openStopValveDialog({
      player: pick.player,
      reasons,
      hasHardPass,
      trigger: button,
    });
  });
  container.append(button);
}

function renderStopValves(cell, pick) {
  const hardPassReasons = pick.hard_pass_reasons || [];
  const concerns = pick.concerns || [];
  const displayedReasons = hardPassReasons.length ? hardPassReasons : concerns;

  const riskLabel = document.createElement("span");
  riskLabel.className = "risk-label";
  riskLabel.textContent = hardPassReasons.length ? "Stop Valves" : (concerns.length ? "Watch" : "Clean");
  cell.append(riskLabel);

  if (displayedReasons.length) {
    const riskList = document.createElement("div");
    riskList.className = "risk-list";
    const tooltip = displayedReasons.join("\n");
    const hasHardPass = hardPassReasons.length > 0;
    riskList.title = tooltip;

    const item = document.createElement("span");
    item.className = hasHardPass ? "risk-reason hard" : "risk-reason concern";
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

    appendStopValveDetailsButton(riskList, pick, displayedReasons, hasHardPass, tooltip);
    cell.append(riskList);
    return;
  }

  const riskText = document.createElement("span");
  riskText.className = "risk-text";
  riskText.textContent = "No stop valves";
  cell.append(riskText);
}

function renderCongregationStatus(cell, value) {
  const congregationStatus = String(value || "").trim();
  if (!congregationStatus) {
    cell.textContent = "--";
    return;
  }

  const statusBadge = document.createElement("span");
  statusBadge.className = `congregation-status status-${congregationStatus.toLowerCase()}`;
  statusBadge.textContent = congregationStatus;
  cell.append(statusBadge);
}

function clearDailyBrief() {
  const brief = document.getElementById("daily-brief");
  if (brief) brief.replaceChildren();
}

function renderReasonList(title, reasons) {
  const section = document.createElement("div");
  const label = document.createElement("strong");
  label.textContent = title;
  section.append(label);

  const list = document.createElement("ul");
  (reasons || []).slice(0, 3).forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    list.append(item);
  });
  section.append(list);
  return section;
}

function renderCommitteeParagraph(title, text) {
  const block = document.createElement("div");
  block.className = "committee-block";
  const label = document.createElement("strong");
  label.textContent = title;
  const paragraph = document.createElement("p");
  paragraph.textContent = text || "No thesis note available.";
  block.append(label, paragraph);
  return block;
}

function renderCommitteeThesis(container, thesis) {
  if (!thesis) return;
  const players = thesis.players || [];
  if (!players.length && !thesis.committee_summary) return;

  const section = document.createElement("section");
  section.className = "committee-thesis";

  const header = document.createElement("div");
  header.className = "committee-thesis-header";
  const titleWrap = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = "Top 2 Committee";
  const subtitle = document.createElement("p");
  subtitle.textContent = thesis.committee_pick
    ? `Single pick: ${thesis.committee_pick}`
    : "Committee notes for the learned top 2.";
  titleWrap.append(title, subtitle);
  header.append(titleWrap);
  section.append(header);

  const grid = document.createElement("div");
  grid.className = "committee-grid";
  players.forEach((player) => {
    const card = document.createElement("article");
    card.className = "committee-card";
    if (player.player && player.player === thesis.committee_pick) {
      card.classList.add("committee-pick");
    }

    const cardHeader = document.createElement("div");
    cardHeader.className = "committee-card-header";
    const nameWrap = document.createElement("div");
    const name = document.createElement("h4");
    name.textContent = player.player || "Top 2 candidate";
    const rank = document.createElement("span");
    rank.textContent = player.rank ? `Learned rank ${player.rank}` : "Learned top 2";
    nameWrap.append(name, rank);
    const badge = document.createElement("strong");
    badge.textContent = player.player === thesis.committee_pick ? "Committee Pick" : "Runner-up";
    cardHeader.append(nameWrap, badge);

    card.append(
      cardHeader,
      renderCommitteeParagraph("Pro Thesis", player.pro_thesis),
      renderCommitteeParagraph("Con Thesis", player.con_thesis),
      renderCommitteeParagraph("Committee Read", player.committee_thesis || thesis.committee_summary)
    );
    grid.append(card);
  });
  section.append(grid);

  if (thesis.source_notes?.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "committee-sources";
    const sourceLabel = document.createElement("strong");
    sourceLabel.textContent = "Sources reviewed";
    sourceList.append(sourceLabel);
    thesis.source_notes.forEach((source) => {
      const link = document.createElement("a");
      link.href = source.url || "#";
      link.textContent = source.label || source.url || "Source";
      link.target = "_blank";
      link.rel = "noreferrer";
      if (source.note) link.title = source.note;
      sourceList.append(link);
    });
    section.append(sourceList);
  }

  container.append(section);
}

function renderDailyBrief(data) {
  const brief = document.getElementById("daily-brief");
  if (!brief) return;
  brief.replaceChildren();

  const selection = data.daily_selection_brief;
  const top2Thesis = data.learned_top2_thesis;
  if ((!selection || !selection.items?.length) && !top2Thesis) return;

  if (selection?.items?.length) {
  const header = document.createElement("div");
  header.className = "daily-brief-header";
  const title = document.createElement("h3");
  title.textContent = "Daily Pick Brief";
  const headline = document.createElement("p");
  headline.textContent = selection.headline || "Review the learned top 5 before making today's contest pick.";
  header.append(title, headline);
  brief.append(header);

  const recommendation = document.createElement("div");
  recommendation.className = "brief-recommendation";
  const single = selection.recommended_single;
  const pair = selection.recommended_pair || [];
  const singleText = single ? `Best 1: ${single.player}` : "Best 1: --";
  const pairText = pair.length
    ? `Best 2: ${pair.map((pick) => pick.player).join(" + ")}`
    : "Best 2: --";
  [singleText, pairText].forEach((text) => {
    const chip = document.createElement("span");
    chip.textContent = text;
    recommendation.append(chip);
  });
  brief.append(recommendation);

  const grid = document.createElement("div");
  grid.className = "brief-grid";
  selection.items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "brief-card";
    if (item.recommendation === "Best 1") card.classList.add("brief-best");
    const cardHeader = document.createElement("div");
    cardHeader.className = "brief-card-header";
    const name = document.createElement("h4");
    name.textContent = item.player;
    const badge = document.createElement("span");
    badge.textContent = item.recommendation || "Watch";
    cardHeader.append(name, badge);

    const score = document.createElement("p");
    score.className = "brief-score";
    score.textContent = `Selection ${numberText(item.selection_score, 1)} · H2H ${item.h2h_record || "0-0"}`;

    card.append(cardHeader, score, renderReasonList("Pros", item.pros), renderReasonList("Cons", item.cons));
    grid.append(card);
  });
  brief.append(grid);
  }

  renderCommitteeThesis(brief, top2Thesis);
}

function renderPick(pick) {
  const template = document.getElementById("learned-pick-template");
  const row = template.content.firstElementChild.cloneNode(true);
  row.classList.add(`result-${resultClass(pick)}`);

  const rank = pick.learned_rank || pick.rank || "";
  row.querySelector(".rank-cell").textContent = rank;
  row.querySelector(".player-name").textContent = pick.player || "Unknown player";
  row.querySelector(".season-ba").textContent = `BA ${formatBattingAverage(pick.hitter_ba_season)}`;
  row.querySelector(".h2h-cell").textContent = pick.h2h_record || "0-0";
  renderCongregationStatus(row.querySelector(".status-cell"), pick.congregation_status);
  row.querySelector(".matchup-line").textContent =
    `${pick.team || "TBD"} vs ${pick.opponent || "TBD"} - ${pick.batter_stand || "?"} bat vs ${pick.pitcher_hand || "?"} arm`;
  row.querySelector(".pitcher-line").textContent = `Probable starter: ${pick.probable_pitcher || "TBD"}`;
  row.querySelector(".venue-line").textContent = `Ballpark: ${pick.venue_name || "TBD"}`;
  row.querySelector(".game-detail-line").textContent = `Start: ${formatGameStart(pick.game_start_time_utc)} \u00B7 ${weatherDetail(pick)}`;

  const badge = row.querySelector(".pick-badge");
  badge.textContent = pick.pickable ? "Pickable" : "Pass";
  badge.classList.toggle("pickable", pick.pickable);

  if (pick.hot_streak) {
    const hotStreak = document.createElement("span");
    hotStreak.className = "hot-streak";
    hotStreak.textContent = "\u{1F525}";
    hotStreak.title = pick.hot_streak_tooltip || "";
    hotStreak.setAttribute("aria-label", `Last 5 games: ${pick.hot_streak_tooltip}`);
    row.querySelector(".player-line").append(hotStreak);
  }

  row.querySelector(".prob-cell strong").textContent = percent(pick.learned_hit_probability);
  row.querySelector(".prob-cell span").textContent = `Bob ${numberText(pick.bob_score ?? pick.score, 1)}`;

  const factorCell = row.querySelector(".factor-cell");
  selectedFactors(pick).forEach((item) => factorCell.append(createFactorChip(item)));
  if (!factorCell.childElementCount) {
    factorCell.append(createChip("No factor detail"));
  }

  renderStopValves(row.querySelector(".risk-cell"), pick);

  const resultCell = row.querySelector(".result-cell");
  const resultBadge = document.createElement("span");
  resultBadge.className = `result-badge result-${resultClass(pick)}`;
  resultBadge.textContent = resultLabel(pick);
  resultCell.append(resultBadge);
  const detail = resultDetail(pick);
  if (detail) {
    const small = document.createElement("small");
    small.textContent = detail;
    resultCell.append(small);
  }

  return row;
}

function renderSummary(data) {
  const picks = data.picks || [];
  const top = picks[0] || {};
  const topProbability = Number(top.learned_hit_probability || 0);
  const probabilityArc = Number.isFinite(topProbability) ? Math.round(topProbability * 360) : 0;

  document.documentElement.style.setProperty("--pickable-arc", `${probabilityArc}deg`);
  setText("top-probability-orbit", picks.length ? percent(top.learned_hit_probability, 0) : "--");
  setText("total-predictions", data.total_predictions ?? "--");
  setText("shown-count", picks.length);
  setText("top-probability", picks.length ? percent(top.learned_hit_probability) : "--");
  setText("top-safety", picks.length ? `${top.safety_score}/100` : "--");
  setText("top-five-result", topFiveResult(picks));

  const boardDate = formatDate(data.date);
  const fallbackText = data.used_latest_fallback
    ? ` Requested ${formatDate(data.requested_date)}, so the latest available learned board is shown.`
    : "";
  setText("board-subtitle", `${boardDate} - Ranked by learned hit probability.${fallbackText}`);

  const modelVersion = data.model_version || "Unknown model";
  const trainedAt = data.model_trained_at ? ` trained ${formatShortDate(data.model_trained_at.slice(0, 10))}` : "";
  setText("status-note", `${modelVersion}${trainedAt}`);
}

function renderBoard(data) {
  renderSummary(data);
  renderDailyBrief(data);
  const list = document.getElementById("picks");
  list.replaceChildren();

  if (!data.picks || !data.picks.length) {
    clearDailyBrief();
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No learned-model predictions available for this date. Run the learned model and export the learned dashboard.";
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
    clearDailyBrief();
    const list = document.getElementById("picks");
    list.innerHTML = `<p class="empty-state">Could not load the selected learned dashboard. Re-run the learned web export, then refresh.</p>`;
    setText("board-subtitle", "Data file not loaded.");
    setText("status-note", "Run the learned dashboard export first.");
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
      clearDailyBrief();
      const list = document.getElementById("picks");
      list.innerHTML = `<p class="empty-state">Could not load ${FALLBACK_DATA_URL}. Start the local web server from the Statbirt folder, then refresh.</p>`;
      setText("board-subtitle", "Data file not loaded.");
      setText("status-note", "Run the learned dashboard export first.");
    }
  }
}

document.getElementById("dashboard-date")?.addEventListener("change", (event) => {
  loadDashboardDate(event.target.value);
});

loadBoard();
