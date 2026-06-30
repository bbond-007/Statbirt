const INDEX_URL = "data/learned_dashboard_index.json";
const FALLBACK_DATA_URL = "data/learned_shortlist.json";
const REPORT_URL = "data/selection_strategy_report.json";

let dashboardIndex = null;
let strategyReport = null;

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
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

function formatBattingAverage(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "--";
  return parsed.toFixed(3).replace(/^0/, "");
}

function normalizeList(value) {
  if (Array.isArray(value)) return value.map(String).map((item) => item.trim()).filter(Boolean);
  return String(value || "")
    .split("|")
    .map((item) => item.trim())
    .filter(Boolean);
}

function isHit(pick) {
  const gameState = String(pick.game_state || "").toLowerCase();
  return gameState === "hit" || pick.result_hit === true || String(pick.result_hit || "").trim() === "1" || Number(pick.result_hits || pick.game_hits || 0) > 0;
}

function finalGameState(pick) {
  const gameStatus = String(pick.game_status || "").toLowerCase();
  return gameStatus.includes("final") || gameStatus.includes("game over") || gameStatus.includes("completed");
}

function effectiveResultStatus(pick) {
  const status = String(pick.result_status || "").toLowerCase();
  const gameState = String(pick.game_state || "").toLowerCase();
  if (status) return status;
  if (gameState === "postponed") return "postponed";
  if (gameState === "final_no_hit") return "final";
  if (gameState === "hit" && finalGameState(pick)) return "final";
  return "pending";
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

function strategyBestReport() {
  return strategyReport?.strategy_summaries?.[0] || null;
}

function strategyBestTop1() {
  return strategyReport?.best_top1_strategy || strategyBestReport();
}

function strategyBestTop2() {
  return strategyReport?.best_top2_strategy || strategyBestReport();
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

function createMetric(label, value) {
  const metric = document.createElement("span");
  metric.className = "strategy-metric";
  const metricLabel = document.createElement("small");
  metricLabel.textContent = label;
  const metricValue = document.createElement("strong");
  metricValue.textContent = value;
  metric.append(metricLabel, metricValue);
  return metric;
}

function createReasonList(title, reasons) {
  const section = document.createElement("div");
  section.className = "strategy-reason-list";
  const label = document.createElement("strong");
  label.textContent = title;
  const list = document.createElement("ul");
  (reasons || []).forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    list.append(item);
  });
  section.append(label, list);
  return section;
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
  closeButton.textContent = "x";
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

function openStopValveDialog({ player, reasons, trigger }) {
  const dialog = ensureStopValveDialog();
  stopValveReturnFocus = trigger || document.activeElement;

  const title = dialog.querySelector("#stop-dialog-title");
  const subtitle = dialog.querySelector(".stop-dialog-subtitle");
  const list = dialog.querySelector(".stop-dialog-list");
  const closeButton = dialog.querySelector(".stop-dialog-close");
  const displayReasons = reasons.length ? reasons : ["No stop valves for this pick."];

  title.textContent = `Stop Valves: ${player || "Candidate"}`;
  subtitle.textContent = reasons.length
    ? `${reasons.length} ${reasons.length === 1 ? "item" : "items"}`
    : "Clean";
  list.replaceChildren();

  displayReasons.forEach((reason) => {
    const item = document.createElement("li");
    item.className = reasons.length ? "hard" : "concern";
    item.textContent = reason;
    list.append(item);
  });

  dialog.hidden = false;
  document.body.classList.add("dialog-open");
  closeButton.focus();
}

function createStopValveButton(item) {
  const reasons = normalizeList(item.hard_pass_reasons);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "strategy-stop-button";
  button.textContent = reasons.length ? `Stop valves (${reasons.length})` : "No stop valves";
  button.setAttribute("aria-label", `Show all stop valves for ${item.player || "candidate"}`);
  button.addEventListener("click", () => {
    openStopValveDialog({
      player: item.player,
      reasons,
      trigger: button,
    });
  });
  return button;
}

function recommendationCard(title, item, extraClass = "") {
  const card = document.createElement("article");
  card.className = `strategy-recommendation ${extraClass}`.trim();
  const label = document.createElement("span");
  label.textContent = title;
  const name = document.createElement("strong");
  name.textContent = item?.player || "--";
  const detail = document.createElement("p");
  detail.textContent = item
    ? `${numberText(item.selection_score, 1)} selection score - ${percent(item.learned_hit_probability)} probability`
    : "No recommendation available";
  card.append(label, name, detail);
  return card;
}

function renderRecommendations(selection) {
  const container = document.getElementById("strategy-recommendations");
  container.replaceChildren();
  const single = selection?.recommended_single || null;
  const pair = selection?.recommended_pair || [];
  container.append(recommendationCard("Best single", single, "primary"));

  const pairCard = document.createElement("article");
  pairCard.className = "strategy-recommendation";
  const pairLabel = document.createElement("span");
  pairLabel.textContent = "Best pair";
  const pairNames = document.createElement("strong");
  pairNames.textContent = pair.length ? pair.map((item) => item.player).join(" + ") : "--";
  const pairDetail = document.createElement("p");
  pairDetail.textContent = pair.length
    ? pair.map((item) => `${item.player}: ${numberText(item.selection_score, 1)}`).join(" - ")
    : "No pair recommendation available";
  pairCard.append(pairLabel, pairNames, pairDetail);
  container.append(pairCard);
}

function renderStrategyCard(item) {
  const card = document.createElement("article");
  card.className = `strategy-card result-${resultClass(item)}`;
  if (item.recommendation === "Best 1") card.classList.add("strategy-best");

  const header = document.createElement("div");
  header.className = "strategy-card-header";
  const title = document.createElement("div");
  const rank = document.createElement("span");
  rank.textContent = `#${item.rank}`;
  const name = document.createElement("h3");
  name.textContent = item.player;
  title.append(rank, name);
  const badge = document.createElement("strong");
  badge.textContent = item.recommendation || item.confidence_label || "Watch";
  header.append(title, badge);

  const thesis = document.createElement("p");
  thesis.className = "strategy-thesis";
  thesis.textContent = item.thesis || item.summary || "No thesis text available.";

  const metrics = document.createElement("div");
  metrics.className = "strategy-metrics";
  metrics.append(
    createMetric("Selection", numberText(item.selection_score, 1)),
    createMetric("Probability", percent(item.learned_hit_probability)),
    createMetric("Safety", item.safety_score == null ? "--" : `${item.safety_score}/100`),
    createMetric("Bob", numberText(item.bob_score, 1)),
    createMetric("H2H", item.h2h_record || "0-0"),
    createMetric("BA", formatBattingAverage(item.hitter_ba_season)),
    createMetric("Expected PA", numberText(item.expected_pa, 1)),
    createMetric("Lineup", numberText(item.lineup_slot, 0))
  );

  const reasons = document.createElement("div");
  reasons.className = "strategy-reason-grid";
  reasons.append(createReasonList("Pros", item.pros), createReasonList("Cons", item.cons));

  const actions = document.createElement("div");
  actions.className = "strategy-card-actions";
  const resultBadge = document.createElement("span");
  resultBadge.className = `result-badge result-${resultClass(item)}`;
  resultBadge.textContent = resultLabel(item);
  actions.append(resultBadge, createStopValveButton(item));
  const detail = resultDetail(item);
  if (detail) {
    const resultSmall = document.createElement("small");
    resultSmall.textContent = detail;
    actions.append(resultSmall);
  }

  card.append(header, thesis, metrics, actions, reasons);
  return card;
}

function renderResearchPanel() {
  const panel = document.getElementById("research-panel");
  panel.replaceChildren();
  if (!strategyReport) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No strategy research report has been generated yet.";
    panel.append(empty);
    return;
  }

  const bestTop1 = strategyBestTop1();
  const bestTop2 = strategyBestTop2();
  const overview = strategyReport.overview || {};
  const header = document.createElement("div");
  header.className = "strategy-research-header";
  const title = document.createElement("h3");
  title.textContent = "Research Backing";
  const detail = document.createElement("p");
  detail.textContent = bestTop1 && bestTop2
    ? `Single: ${bestTop1.label} at ${percent(bestTop1.top1_hit_rate)}. Pair: ${bestTop2.label} at ${percent(bestTop2.top2_any_hit_rate)} any-hit.`
    : "Research summary loaded.";
  const titleBlock = document.createElement("div");
  titleBlock.append(title, detail);
  header.append(titleBlock);
  if (strategyReport.web_report_url) {
    const reportLink = document.createElement("a");
    reportLink.className = "strategy-report-link";
    reportLink.href = strategyReport.web_report_url;
    reportLink.textContent = "6-19-26 Strategy Report";
    header.append(reportLink);
  }

  const grid = document.createElement("div");
  grid.className = "strategy-metrics research-metrics";
  grid.append(
    createMetric("Candidate decisions", overview.labeled_decisions?.toLocaleString?.() || "--"),
    createMetric("Date range", overview.date_min && overview.date_max ? `${overview.date_min} to ${overview.date_max}` : "--"),
    createMetric("Overall hit rate", percent(overview.overall_hit_rate)),
    createMetric("Best top-2 streak", bestTop2 ? bestTop2.top2_any_max_streak : "--")
  );
  panel.append(header, grid);
}

function renderSummary(data) {
  const selection = data.daily_selection_brief || {};
  const single = selection.recommended_single || {};
  const pair = selection.recommended_pair || [];
  const bestTop1 = strategyBestTop1();
  const bestTop2 = strategyBestTop2();
  const topScore = Number(single.selection_score || 0);
  const scoreArc = Number.isFinite(topScore) ? Math.round((topScore / 100) * 360) : 0;

  document.documentElement.style.setProperty("--pickable-arc", `${scoreArc}deg`);
  setText("selection-score-orbit", single.selection_score ? numberText(single.selection_score, 0) : "--");
  setText("best-single", single.player || "--");
  setText("best-pair", pair.length ? pair.map((item) => item.player).join(" + ") : "--");
  setText("top1-backtest", bestTop1 ? percent(bestTop1.top1_hit_rate) : "--");
  setText("top2-backtest", bestTop2 ? percent(bestTop2.top2_any_hit_rate) : "--");
  setText("report-decisions", strategyReport?.overview?.labeled_decisions?.toLocaleString?.() || "--");

  setText("strategy-headline", selection.headline || "Review the learned top 5 before making today's contest pick.");
  setText("board-subtitle", `${formatDate(data.date)} - One- or two-hitter contest selection strategy.`);
  const modelVersion = data.model_version || "Unknown model";
  const trainedAt = data.model_trained_at ? ` trained ${formatShortDate(data.model_trained_at.slice(0, 10))}` : "";
  setText("status-note", `${modelVersion}${trainedAt}`);
}

function renderStrategy(data) {
  renderSummary(data);
  const selection = data.daily_selection_brief;
  const cards = document.getElementById("strategy-cards");
  const recommendations = document.getElementById("strategy-recommendations");
  cards.replaceChildren();
  recommendations.replaceChildren();
  if (!selection || !selection.items?.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No learned selection brief is available for this date.";
    cards.append(empty);
    renderResearchPanel();
    return;
  }
  renderRecommendations(selection);
  selection.items.forEach((item) => cards.append(renderStrategyCard(item)));
  renderResearchPanel();
}

async function loadDashboardDate(date) {
  try {
    const entry = dashboardEntryByDate(date);
    const data = await fetchJson(dashboardDataUrl(entry));
    renderStrategy(data);
    const select = document.getElementById("dashboard-date");
    if (select && data.date) select.value = data.date;
  } catch (error) {
    const cards = document.getElementById("strategy-cards");
    cards.innerHTML = `<p class="empty-state">Could not load the selected strategy data. Re-run the learned web export, then refresh.</p>`;
    setText("board-subtitle", "Data file not loaded.");
    setText("status-note", "Run the learned dashboard export first.");
  }
}

async function loadStrategyReport() {
  try {
    strategyReport = await fetchJson(REPORT_URL);
  } catch (error) {
    strategyReport = null;
  }
}

async function loadBoard() {
  await loadStrategyReport();
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
      renderStrategy(data);
    } catch (fallbackError) {
      const cards = document.getElementById("strategy-cards");
      cards.innerHTML = `<p class="empty-state">Could not load ${FALLBACK_DATA_URL}. Start the local web server from the Statbirt folder, then refresh.</p>`;
      setText("board-subtitle", "Data file not loaded.");
      setText("status-note", "Run the learned dashboard export first.");
    }
  }
}

document.getElementById("dashboard-date")?.addEventListener("change", (event) => {
  loadDashboardDate(event.target.value);
});

loadBoard();
