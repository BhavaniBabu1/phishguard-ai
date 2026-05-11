/**
 * script.js — PhishGuard AI Frontend
 * =====================================
 * Handles URL scanning, result rendering, history,
 * model stats fetch, and all UI interactions.
 */

"use strict";

// ─── DOM refs ──────────────────────────────────────────────────────────────
const urlInput        = document.getElementById("urlInput");
const inputWrap       = document.getElementById("inputWrap");
const inputError      = document.getElementById("inputError");
const scanBtn         = document.getElementById("scanBtn");
const clearBtn        = document.getElementById("clearBtn");
const loadingPanel    = document.getElementById("loadingPanel");
const resultPanel     = document.getElementById("resultPanel");
const loaderText      = document.getElementById("loaderText");

// Verdict
const verdictIcon     = document.getElementById("verdictIcon");
const verdictLabel    = document.getElementById("verdictLabel");
const verdictUrl      = document.getElementById("verdictUrl");
const verdictBadge    = document.getElementById("verdictBadge");

// Confidence meter
const confScore       = document.getElementById("confScore");
const confFill        = document.getElementById("confFill");
const confThumb       = document.getElementById("confThumb");
const legitProb       = document.getElementById("legitProb");
const phishProb       = document.getElementById("phishProb");

// Features
const featuresGrid    = document.getElementById("featuresGrid");
const riskTag         = document.getElementById("riskTag");

// Stats
const modelAccuracy   = document.getElementById("modelAccuracy");
const modelF1         = document.getElementById("modelF1");
const modelAUC        = document.getElementById("modelAUC");
const sessionScans    = document.getElementById("sessionScans");
const modelStatus     = document.getElementById("modelStatus");

// History
const historyList     = document.getElementById("historyList");
const historyCount    = document.getElementById("historyCount");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");


// ─── State ─────────────────────────────────────────────────────────────────
let scanCount = 0;
let localHistory = [];


// ─── Loader messages ───────────────────────────────────────────────────────
const LOADER_STEPS = [
  "Extracting URL features…",
  "Analyzing domain structure…",
  "Running Random Forest…",
  "Scoring 200 decision trees…",
  "Calculating confidence…",
];
let loaderInterval = null;

function startLoader() {
  let i = 0;
  loaderText.textContent = LOADER_STEPS[0];
  loaderInterval = setInterval(() => {
    i = (i + 1) % LOADER_STEPS.length;
    loaderText.textContent = LOADER_STEPS[i];
  }, 600);
}

function stopLoader() {
  clearInterval(loaderInterval);
}


// ─── UI helpers ────────────────────────────────────────────────────────────

function showPanel(name) {
  loadingPanel.classList.toggle("visible", name === "loading");
  resultPanel.classList.toggle("visible",  name === "result");
}

function showError(msg) {
  inputError.textContent = msg;
  inputWrap.classList.add("error");
}

function clearError() {
  inputError.textContent = "";
  inputWrap.classList.remove("error");
}

function setButtonState(loading) {
  scanBtn.disabled = loading;
  if (loading) {
    scanBtn.querySelector(".scan-btn__text").textContent = "Analyzing…";
    scanBtn.querySelector(".scan-btn__icon").textContent = "⏳";
  } else {
    scanBtn.querySelector(".scan-btn__text").textContent = "Analyze URL";
    scanBtn.querySelector(".scan-btn__icon").textContent = "→";
  }
}

function formatConfidence(phishPct) {
  if (phishPct >= 80) return `${phishPct}% — Critical Risk`;
  if (phishPct >= 60) return `${phishPct}% — High Risk`;
  if (phishPct >= 40) return `${phishPct}% — Medium Risk`;
  return `${100 - phishPct}% — Low Risk`;
}


// ─── Render result ─────────────────────────────────────────────────────────

function renderResult(data) {
  const isPhish = data.prediction === 1;
  const riskCls = `risk--${data.risk_level}`;

  // Verdict
  verdictIcon.textContent = isPhish ? "🚨" : "✅";
  verdictIcon.className   = `verdict__icon ${isPhish ? "phish" : "legit"}`;
  verdictLabel.textContent = data.label;
  verdictLabel.className   = `verdict__label ${isPhish ? "phish" : "legit"}`;
  verdictUrl.textContent   = data.url;

  const riskLabels = { critical: "Critical", high: "High Risk", medium: "Medium", low: "Low Risk" };
  verdictBadge.textContent = riskLabels[data.risk_level] || data.risk_level;
  verdictBadge.className   = `verdict__badge ${riskCls}`;

  // Confidence meter
  const phishPct = data.phish_probability;
  confFill.style.width = `${phishPct}%`;
  confFill.className   = `conf-meter__fill ${isPhish ? "phish" : "legit"}`;
  confThumb.style.left = `${phishPct}%`;
  confScore.textContent = formatConfidence(phishPct);
  legitProb.textContent = `✓ Legit ${data.legit_probability}%`;
  phishProb.textContent = `⚠ Phish ${data.phish_probability}%`;

  // Risk tag
  riskTag.textContent = (riskLabels[data.risk_level] || data.risk_level).toUpperCase();
  riskTag.className   = `risk-tag ${riskCls}`;

  // Features grid
  featuresGrid.innerHTML = "";
  data.features.forEach((f, i) => {
    const item = document.createElement("div");
    item.className = "feature-item";
    item.style.animationDelay = `${i * 40}ms`;

    item.innerHTML = `
      <div class="feature-dot risk--${f.risk}"></div>
      <div class="feature-info">
        <div class="feature-name">${escapeHtml(f.name)}</div>
        <div class="feature-value">${escapeHtml(String(f.value))}</div>
      </div>
    `;
    item.title = f.description;
    featuresGrid.appendChild(item);
  });

  showPanel("result");
}


// ─── Feature display ───────────────────────────────────────────────────────

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}


// ─── Scan ──────────────────────────────────────────────────────────────────

async function scan(rawUrl) {
  const url = rawUrl.trim();
  if (!url) { showError("Please enter a URL."); return; }
  clearError();
  setButtonState(true);
  showPanel("loading");
  startLoader();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    const data = await res.json();
    stopLoader();

    if (!res.ok) {
      showError(data.error || "Prediction failed. Try again.");
      showPanel(null);
      return;
    }

    renderResult(data);
    addToHistory(data);
    scanCount++;
    sessionScans.textContent = scanCount;
    updateSessionStats();

  } catch (err) {
    stopLoader();
    showError("Network error. Is the Flask server running?");
    showPanel(null);
  } finally {
    setButtonState(false);
  }
}


// ─── History ───────────────────────────────────────────────────────────────

function addToHistory(data) {
  localHistory.unshift(data);
  if (localHistory.length > 50) localHistory.pop();
  renderHistory();
}

function renderHistory() {
  if (localHistory.length === 0) {
    historyList.innerHTML = `<div class="history-empty">No scans yet. Analyze a URL to get started.</div>`;
    historyCount.textContent = "0";
    return;
  }
  historyCount.textContent = localHistory.length;
  historyList.innerHTML = localHistory.map(d => `
    <div class="history-item ${d.prediction === 1 ? "phish" : "legit"}">
      <span class="history-label ${d.prediction === 1 ? "phish" : "legit"}">${d.label}</span>
      <span class="history-url" title="${escapeHtml(d.url)}">${escapeHtml(d.url)}</span>
      <span class="history-conf">${d.confidence}%</span>
      <span class="history-time">${d.timestamp || ""}</span>
    </div>
  `).join("");
}

clearHistoryBtn.addEventListener("click", () => {
  localHistory = [];
  scanCount = 0;
  sessionScans.textContent = "0";
  renderHistory();
});


// ─── Model info ────────────────────────────────────────────────────────────

async function loadModelInfo() {
  try {
    const res  = await fetch("/api/model-info");
    const data = await res.json();
    if (data.accuracy)  modelAccuracy.textContent = `${data.accuracy}%`;
    if (data.f1_score)  modelF1.textContent        = `${data.f1_score}%`;
    if (data.roc_auc)   modelAUC.textContent        = `${data.roc_auc}%`;
  } catch (_) {
    modelStatus.innerHTML = `<span class="badge__dot" style="background:#ff4d6d"></span>Model Offline`;
  }
}


async function updateSessionStats() {
  try {
    await fetch("/api/stats");
  } catch (_) {}
}


// ─── Event listeners ───────────────────────────────────────────────────────

// Input typing
urlInput.addEventListener("input", () => {
  clearError();
  clearBtn.classList.toggle("visible", urlInput.value.length > 0);
});

// Enter key
urlInput.addEventListener("keydown", e => {
  if (e.key === "Enter") scan(urlInput.value);
});

// Scan button
scanBtn.addEventListener("click", () => scan(urlInput.value));

// Clear
clearBtn.addEventListener("click", () => {
  urlInput.value = "";
  clearBtn.classList.remove("visible");
  clearError();
  showPanel(null);
  urlInput.focus();
});

// Example pills
document.querySelectorAll(".example-pill").forEach(pill => {
  pill.addEventListener("click", () => {
    const url = pill.dataset.url;
    urlInput.value = url;
    clearBtn.classList.add("visible");
    clearError();
    scan(url);
  });
});


// ─── Init ──────────────────────────────────────────────────────────────────

(function init() {
  loadModelInfo();
  renderHistory();

  // Stagger stat cards
  document.querySelectorAll(".stat-card").forEach((card, i) => {
    card.style.opacity = "0";
    card.style.transform = "translateY(16px)";
    setTimeout(() => {
      card.style.transition = "opacity .4s ease, transform .4s ease";
      card.style.opacity = "1";
      card.style.transform = "translateY(0)";
    }, 200 + i * 80);
  });
})();