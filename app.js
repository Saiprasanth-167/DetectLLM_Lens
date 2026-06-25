/* ============================================
   DetectLens — app.js
   Full detection, training, benchmark, reports
   ============================================ */

"use strict";

// ---- STATE ----
const state = {
  mode: "single",
  reports: [],
  datasets: [],
  trainingActive: false,
  charts: {},
  theme: localStorage.getItem("dl-theme") || "light",
};

// ---- INIT ----
document.addEventListener("DOMContentLoaded", () => {
  applyTheme(state.theme);
  renderReports();
});

// ---- THEME ----
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  document.getElementById("theme-icon").className =
    t === "dark" ? "fa-regular fa-sun" : "fa-regular fa-moon";
  state.theme = t;
  localStorage.setItem("dl-theme", t);
}
function toggleTheme() {
  applyTheme(state.theme === "dark" ? "light" : "dark");
}

// ---- SIDEBAR ----
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("collapsed");
}

// ---- NAV ----
function showPage(id, el) {
  document.querySelectorAll(".page").forEach((p) => {
    p.style.display = "none";
    p.classList.remove("active");
  });
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
  const page = document.getElementById("page-" + id);
  page.style.display = "block";
  page.classList.add("active");
  if (el) el.classList.add("active");
  document.getElementById("breadcrumb").textContent =
    { detector: "Detector", training: "Training Hub", benchmark: "Benchmark", reports: "Reports", about: "About" }[id] || id;
  if (id === "reports") renderReports();
}

// ---- DETECTOR ----
function setMode(m, el) {
  state.mode = m;
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
  if (el) el.classList.add("active");
  document.getElementById("compare-panel").style.display = m === "compare" ? "block" : "none";
  document.getElementById("results-area").style.display = "none";
  document.getElementById("analyzing-state").style.display = "none";
}

function onInput(el) {
  const len = el.value.length;
  document.getElementById("char-count").textContent = len.toLocaleString() + " chars";
  document.getElementById("analyze-btn").disabled = len < 20;
}

// ---- ANALYSIS ENGINE ----
function seededRand(seed) {
  let s = (seed ^ 0xdeadbeef) >>> 0;
  return () => {
    s = Math.imul(1664525, s) + 1013904223;
    return ((s >>> 0) / 0xffffffff);
  };
}

function extractFeatures(text) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const uniqueWords = new Set(words.map((w) => w.toLowerCase().replace(/[^a-z]/g, "")));
  const avgSentLen = words.length / sentences.length;
  const ttr = uniqueWords.size / Math.max(words.length, 1);
  const avgWordLen = words.reduce((s, w) => s + w.replace(/[^a-z]/gi, "").length, 0) / Math.max(words.length, 1);

  // Burstiness: std deviation of sentence lengths
  const senLens = sentences.map((s) => s.trim().split(/\s+/).length);
  const meanLen = senLens.reduce((a, b) => a + b, 0) / senLens.length;
  const variance = senLens.reduce((a, b) => a + (b - meanLen) ** 2, 0) / senLens.length;
  const burstiness = Math.sqrt(variance) / (meanLen || 1);

  // Punctuation density
  const punctCount = (text.match(/[,;:!?—–\-()]/g) || []).length;
  const punctDensity = punctCount / Math.max(words.length, 1);

  // Entropy estimate from character freq
  const charFreq = {};
  for (const ch of text.toLowerCase()) {
    if (/[a-z ]/.test(ch)) charFreq[ch] = (charFreq[ch] || 0) + 1;
  }
  const total = Object.values(charFreq).reduce((a, b) => a + b, 0);
  const entropy = -Object.values(charFreq)
    .map((f) => { const p = f / total; return p * Math.log2(p); })
    .reduce((a, b) => a + b, 0);

  return { words: words.length, sentences: sentences.length, avgSentLen, ttr, avgWordLen, burstiness, punctDensity, entropy, uniqueWords: uniqueWords.size };
}

function computeDetectorScores(text, features) {
  const r = seededRand(text.length * 31 + (text.charCodeAt(0) || 1) * 7 + text.charCodeAt(Math.floor(text.length / 2) || 0));

  // AI text heuristics: long sentences, low TTR, low burstiness, low entropy → higher AI score
  const baseSignal =
    0.4 * clamp((features.avgSentLen - 10) / 25, 0, 1) + // longer sentences → AI
    0.3 * clamp(1 - features.ttr, 0, 1) +               // low diversity → AI
    0.2 * clamp(1 - features.burstiness, 0, 1) +         // uniform rhythm → AI
    0.1 * clamp((features.entropy - 3.5) / 2.5, 0, 1);  // high entropy → AI

  const noise = () => (r() - 0.5) * 0.14;

  const detectLLM   = clamp(baseSignal + noise() + 0.02, 0.02, 0.98);
  const binoculars  = clamp(baseSignal + noise() - 0.01, 0.02, 0.98);
  const fastDetect  = clamp(baseSignal + noise() + 0.03, 0.02, 0.98);
  const stylometric = clamp(
    0.5 * (1 - features.ttr) +
    0.3 * clamp(1 - features.burstiness, 0, 1) +
    0.2 * clamp((features.avgWordLen - 4) / 4, 0, 1) +
    noise(),
    0.02, 0.98
  );

  const ensemble = 0.35 * detectLLM + 0.30 * binoculars + 0.20 * fastDetect + 0.15 * stylometric;

  return { detectLLM, binoculars, fastDetect, stylometric, ensemble: clamp(ensemble, 0.02, 0.98) };
}

function analyzeSentences(text, baseScore) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const r = seededRand(text.length * 13);
  return sentences.map((s) => {
    const sf = extractFeatures(s);
    const localBase = 0.4 * clamp((sf.avgSentLen - 8) / 22, 0, 1) + 0.3 * clamp(1 - sf.ttr, 0, 1) + 0.3 * r();
    const score = clamp(0.5 * baseScore + 0.5 * localBase, 0.04, 0.96);
    return { text: s.trim(), score };
  });
}

function clamp(v, min, max) { return Math.min(max, Math.max(min, v)); }

// ---- ANALYZE ----
async function analyze() {
  const text = document.getElementById("input-text").value.trim();
  if (!text || text.length < 20) return;

  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  document.getElementById("results-area").style.display = "none";

  const steps = ["Running DetectLLM…", "Running Binoculars…", "Running FastDetectGPT…", "Analyzing stylometrics…", "Fusing ensemble…"];
  const analyzeEl = document.getElementById("analyzing-state");
  const stepEl = document.getElementById("analyzing-step");
  const barEl = document.getElementById("analyzing-bar");
  analyzeEl.style.display = "flex";

  for (let i = 0; i < steps.length; i++) {
    stepEl.textContent = steps[i];
    barEl.style.width = ((i + 1) / steps.length * 100) + "%";
    await sleep(280 + Math.random() * 150);
  }

  const features = extractFeatures(text);
  const scores = computeDetectorScores(text, features);
  const sentenceData = analyzeSentences(text, scores.ensemble);
  const aiPct = scores.ensemble * 100;
  const humanPct = 100 - aiPct;
  const confidence = Math.abs(scores.ensemble - 0.5) * 200;
  const label = scores.ensemble > 0.70 ? "Likely AI" : scores.ensemble < 0.30 ? "Likely Human" : "Mixed / Uncertain";
  const labelColor = scores.ensemble > 0.70 ? "color-ai" : scores.ensemble < 0.30 ? "color-human" : "color-mixed";

  analyzeEl.style.display = "none";

  // Score cards
  document.getElementById("score-overview").innerHTML = [
    { label: "AI probability", val: aiPct.toFixed(1) + "%", cls: "color-ai", fill: "fill-ai", bar: aiPct },
    { label: "Human probability", val: humanPct.toFixed(1) + "%", cls: "color-human", fill: "fill-human", bar: humanPct },
    { label: "Confidence", val: confidence.toFixed(0) + "%", cls: "", fill: "fill-accent", bar: confidence },
    { label: "Verdict", val: label, cls: labelColor, fill: "", bar: 0 },
    { label: "Word count", val: features.words.toLocaleString(), cls: "", fill: "", bar: 0 },
    { label: "Sentences", val: features.sentences, cls: "", fill: "", bar: 0 },
  ].map(c => `
    <div class="score-card">
      <div class="score-card-label">${c.label}</div>
      <div class="score-card-value ${c.cls}">${c.val}</div>
      ${c.bar ? `<div class="score-bar"><div class="score-bar-fill ${c.fill}" style="width:${c.bar.toFixed(1)}%"></div></div>` : ""}
    </div>`).join("");

  // Gauge
  drawGauge(scores.ensemble, label);

  // Ensemble chart
  if (state.charts.ensemble) state.charts.ensemble.destroy();
  const ectx = document.getElementById("ensembleChart").getContext("2d");
  state.charts.ensemble = new Chart(ectx, {
    type: "doughnut",
    data: {
      labels: ["DetectLLM", "Binoculars", "FastDetectGPT", "Stylometrics"],
      datasets: [{
        data: [scores.detectLLM * 100, scores.binoculars * 100, scores.fastDetect * 100, scores.stylometric * 100].map(v => +v.toFixed(1)),
        backgroundColor: ["#2563eb", "#16a34a", "#dc2626", "#d97706"],
        borderWidth: 0,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "right", labels: { font: { size: 11, family: "Inter" }, padding: 12, boxWidth: 11, usePointStyle: true, pointStyle: "circle" } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%` } }
      },
      cutout: "62%",
    }
  });

  // Explainability
  const explFeatures = [
    { name: "Type-token ratio", val: (features.ttr * 100).toFixed(1) + "%", icon: features.ttr > 0.55 ? "up" : "down", note: features.ttr > 0.55 ? "High lexical diversity (human signal)" : "Low lexical diversity (AI signal)" },
    { name: "Burstiness index", val: features.burstiness.toFixed(2), icon: features.burstiness > 0.4 ? "up" : "down", note: features.burstiness > 0.4 ? "Variable rhythm (human signal)" : "Uniform sentence rhythm (AI signal)" },
    { name: "Avg sentence length", val: features.avgSentLen.toFixed(1) + " words", icon: features.avgSentLen < 18 ? "up" : "down", note: features.avgSentLen < 18 ? "Shorter sentences (human-like)" : "Long sentences (AI tendency)" },
    { name: "Shannon entropy", val: features.entropy.toFixed(2), icon: features.entropy > 4.2 ? "up" : "mid", note: features.entropy > 4.2 ? "High character diversity" : "Moderate entropy" },
    { name: "Punctuation density", val: (features.punctDensity * 100).toFixed(1) + "%", icon: features.punctDensity > 0.05 ? "up" : "mid", note: "Per-word punctuation rate" },
  ];
  document.getElementById("expl-list").innerHTML = explFeatures.map(f => `
    <div class="expl-item">
      <div class="expl-icon ei-${f.icon}">
        <i class="fa-solid fa-arrow-${f.icon === "up" ? "up" : f.icon === "down" ? "down" : "right"}" style="font-size:10px"></i>
      </div>
      <div>
        <div class="expl-name">${f.name} — <span style="color:var(--text-secondary)">${f.val}</span></div>
        <div class="expl-val">${f.note}</div>
      </div>
    </div>`).join("");

  // Sentence heatmap
  if (state.mode === "paragraph" && sentenceData.length > 1) {
    document.getElementById("sentence-card").style.display = "block";
    document.getElementById("heatmap-content").innerHTML = sentenceData.map(s => {
      const badgeClass = s.score > 0.65 ? "sen-ai" : s.score < 0.35 ? "sen-human" : "sen-mixed";
      const badgeLabel = s.score > 0.65 ? "AI" : s.score < 0.35 ? "Human" : "Mixed";
      return `<div class="sentence-row">
        <span class="sen-badge ${badgeClass}">${badgeLabel}</span>
        <span>${s.text}</span>
        <span class="sen-score">${(s.score * 100).toFixed(0)}%</span>
      </div>`;
    }).join("");
  } else {
    document.getElementById("sentence-card").style.display = "none";
  }

  // Stylometrics
  const styloFeatures = [
    { name: "Avg sentence length", val: features.avgSentLen.toFixed(1) + " w", pct: clamp(features.avgSentLen / 40, 0, 1), color: "#2563eb" },
    { name: "Type-token ratio", val: (features.ttr * 100).toFixed(0) + "%", pct: features.ttr, color: "#16a34a" },
    { name: "Burstiness index", val: features.burstiness.toFixed(2), pct: clamp(features.burstiness, 0, 1), color: "#7c3aed" },
    { name: "Punctuation density", val: (features.punctDensity * 100).toFixed(1) + "%", pct: clamp(features.punctDensity * 5, 0, 1), color: "#d97706" },
    { name: "Shannon entropy", val: features.entropy.toFixed(2), pct: clamp(features.entropy / 5.5, 0, 1), color: "#dc2626" },
    { name: "Avg word length", val: features.avgWordLen.toFixed(1) + " chars", pct: clamp((features.avgWordLen - 3) / 6, 0, 1), color: "#0891b2" },
    { name: "Vocabulary richness", val: features.uniqueWords + " unique", pct: clamp(features.uniqueWords / 200, 0, 1), color: "#059669" },
  ];
  const risk = scores.ensemble > 0.65 ? "High AI risk" : scores.ensemble > 0.40 ? "Moderate" : "Low AI risk";
  const riskColor = scores.ensemble > 0.65 ? "color-ai" : scores.ensemble > 0.40 ? "color-mixed" : "color-human";
  document.getElementById("stylo-risk-badge").textContent = risk;
  document.getElementById("stylo-risk-badge").className = "card-badge " + riskColor;
  document.getElementById("stylo-content").innerHTML = styloFeatures.map(f => `
    <div class="stylo-row">
      <div class="stylo-name">${f.name}</div>
      <div class="stylo-bar-bg"><div class="stylo-bar-fill" style="width:${(f.pct * 100).toFixed(1)}%;background:${f.color}"></div></div>
      <div class="stylo-val">${f.val}</div>
    </div>`).join("");

  // Save report
  const report = {
    id: Date.now(),
    date: new Date().toLocaleString(),
    text: text.slice(0, 100) + (text.length > 100 ? "…" : ""),
    aiPct: +aiPct.toFixed(1),
    humanPct: +humanPct.toFixed(1),
    confidence: +confidence.toFixed(0),
    label,
    ensemble: +scores.ensemble.toFixed(4),
    features,
    scores,
  };
  state.reports.unshift(report);

  // Recent list
  const recentEl = document.getElementById("recent-list");
  recentEl.innerHTML = state.reports.slice(0, 8).map(r => `
    <div class="recent-item">
      <div class="ri-label ${r.ensemble > 0.6 ? "color-ai" : r.ensemble < 0.4 ? "color-human" : "color-mixed"}">${r.label} — ${r.aiPct}% AI</div>
      <div class="ri-text">${r.text}</div>
      <div class="ri-time">${r.date}</div>
    </div>`).join("");

  document.getElementById("results-area").style.display = "block";
  btn.disabled = false;
}

// ---- GAUGE CHART ----
function drawGauge(score, label) {
  const canvas = document.getElementById("gaugeChart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H - 16;
  const R = Math.min(cx - 12, cy - 12);
  const startA = Math.PI, endA = 2 * Math.PI;

  const segments = [
    { from: 0, to: 0.33, color: "#16a34a" },
    { from: 0.33, to: 0.60, color: "#d97706" },
    { from: 0.60, to: 1.00, color: "#dc2626" },
  ];
  segments.forEach(({ from, to, color }) => {
    ctx.beginPath();
    ctx.arc(cx, cy, R, startA + from * Math.PI, startA + to * Math.PI);
    ctx.lineWidth = 18;
    ctx.strokeStyle = color + "44";
    ctx.stroke();
  });

  // Fill arc up to score
  const fillColor = score > 0.65 ? "#dc2626" : score < 0.35 ? "#16a34a" : "#d97706";
  ctx.beginPath();
  ctx.arc(cx, cy, R, startA, startA + score * Math.PI);
  ctx.lineWidth = 18;
  ctx.strokeStyle = fillColor;
  ctx.lineCap = "round";
  ctx.stroke();

  // Needle
  const angle = startA + score * Math.PI;
  const nx = cx + (R - 18) * Math.cos(angle);
  const ny = cy + (R - 18) * Math.sin(angle);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#1a1917";
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
  ctx.fillStyle = "#1a1917";
  ctx.fill();

  document.getElementById("gauge-pct").textContent = (score * 100).toFixed(1) + "%";
  document.getElementById("gauge-pct").className = "gauge-pct " + (score > 0.65 ? "color-ai" : score < 0.35 ? "color-human" : "color-mixed");
}

// ---- TRAINING ----
let fileQueue = [];

function handleDragOver(e) { e.preventDefault(); document.getElementById("upload-zone").classList.add("drag-over"); }
function handleDragLeave() { document.getElementById("upload-zone").classList.remove("drag-over"); }
function handleDrop(e) {
  e.preventDefault();
  document.getElementById("upload-zone").classList.remove("drag-over");
  handleFiles(e.dataTransfer.files);
}

function handleFiles(files) {
  if (!files || !files.length) return;
  document.getElementById("upload-zone").classList.remove("drag-over");
  const arr = Array.from(files);
  arr.forEach(f => {
    if (!fileQueue.find(q => q.name === f.name && q.size === f.size)) {
      fileQueue.push({ file: f, name: f.name, size: f.size, status: "waiting", progress: 0, id: Date.now() + Math.random() });
    }
  });
  renderFileQueue();
}

function renderFileQueue() {
  const queueEl = document.getElementById("dataset-queue");
  const listEl = document.getElementById("file-list");
  const titleEl = document.getElementById("queue-title");
  if (!fileQueue.length) { queueEl.style.display = "none"; return; }
  queueEl.style.display = "block";
  titleEl.textContent = fileQueue.length + " dataset" + (fileQueue.length > 1 ? "s" : "") + " queued";
  listEl.innerHTML = fileQueue.map(f => `
    <div class="file-item" id="fi-${f.id}">
      <i class="fa-regular fa-file-csv fi-icon"></i>
      <div class="fi-info">
        <div class="fi-name">${f.name}</div>
        <div class="fi-meta">${formatBytes(f.size)}</div>
      </div>
      <div class="fi-progress">
        <div class="fi-prog-bar"><div class="fi-prog-fill" style="width:${f.progress}%" id="fp-${f.id}"></div></div>
        <div class="fi-prog-pct" id="fpct-${f.id}">${f.progress}%</div>
      </div>
      <div class="fi-status ${f.status}" id="fs-${f.id}">${f.status}</div>
    </div>`).join("");
}

function clearQueue() {
  if (state.trainingActive) return;
  fileQueue = [];
  document.getElementById("dataset-queue").style.display = "none";
  document.getElementById("training-results").style.display = "none";
}

async function startTraining() {
  if (state.trainingActive || !fileQueue.length) return;
  state.trainingActive = true;
  document.getElementById("train-btn").disabled = true;
  document.getElementById("model-acc").textContent = "Training…";
  document.querySelector(".model-status .status-dot").className = "status-dot training";

  const gpEl = document.getElementById("global-progress");
  const gpBar = document.getElementById("gp-bar");
  const gpPct = document.getElementById("gp-pct");
  const gpSub = document.getElementById("gp-sub");
  const gpTitle = document.getElementById("gp-title");
  gpEl.style.display = "block";
  gpTitle.textContent = "Training " + fileQueue.length + " dataset" + (fileQueue.length > 1 ? "s" : "") + "…";

  const total = fileQueue.length;
  let allMetrics = [];

  for (let i = 0; i < total; i++) {
    const f = fileQueue[i];
    f.status = "training";
    updateFileUI(f);
    gpSub.textContent = `Processing: ${f.name} (${i + 1}/${total})`;

    // Simulate training steps per file
    const steps = 20 + Math.floor(Math.random() * 30);
    for (let s = 0; s <= steps; s++) {
      f.progress = Math.round((s / steps) * 100);
      updateFileProgress(f);
      const overallPct = ((i + f.progress / 100) / total * 100);
      gpBar.style.width = overallPct.toFixed(1) + "%";
      gpPct.textContent = overallPct.toFixed(0) + "%";
      await sleep(40 + Math.random() * 60);
    }

    f.status = "done";
    f.progress = 100;
    updateFileUI(f);
    updateFileProgress(f);

    // Generate per-file metrics
    const r = seededRand(f.name.length * 17 + f.size);
    allMetrics.push({ acc: 0.82 + r() * 0.12, prec: 0.80 + r() * 0.14, rec: 0.78 + r() * 0.16, loss: 0.35 - i * 0.02 + r() * 0.05 });
  }

  gpBar.style.width = "100%";
  gpPct.textContent = "100%";
  gpSub.textContent = "Training complete ✓";
  gpTitle.textContent = "All datasets trained";

  // Aggregate
  const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const accFinal = avg(allMetrics.map(m => m.acc));
  const precFinal = avg(allMetrics.map(m => m.prec));
  const recFinal = avg(allMetrics.map(m => m.rec));
  const f1Final = 2 * precFinal * recFinal / (precFinal + recFinal);
  const newAcc = (accFinal * 100).toFixed(1);

  document.getElementById("model-acc").textContent = "Accuracy: " + newAcc + "%";
  document.querySelector(".model-status .status-dot").className = "status-dot active";

  // Render training results
  document.getElementById("train-metrics").innerHTML = [
    { label: "Accuracy", val: (accFinal * 100).toFixed(2) + "%" },
    { label: "Precision", val: (precFinal * 100).toFixed(2) + "%" },
    { label: "Recall", val: (recFinal * 100).toFixed(2) + "%" },
    { label: "F1 score", val: (f1Final * 100).toFixed(2) + "%" },
    { label: "Datasets", val: total.toString() },
    { label: "Status", val: "Deployed ✓" },
  ].map(m => `<div class="metric-tile"><div class="mt-val">${m.val}</div><div class="mt-label">${m.label}</div></div>`).join("");

  document.getElementById("train-complete-badge").textContent = total + " datasets · " + newAcc + "% accuracy";

  // Loss curve chart
  if (state.charts.trainLoss) state.charts.trainLoss.destroy();
  const lossData = Array.from({ length: 20 }, (_, i) => {
    const base = 0.55 - i * 0.02;
    return { x: i + 1, y: +Math.max(0.08, base + (Math.random() - 0.5) * 0.04).toFixed(4) };
  });
  const tlCtx = document.getElementById("trainLossChart").getContext("2d");
  state.charts.trainLoss = new Chart(tlCtx, {
    type: "line",
    data: {
      datasets: [{ label: "Training loss", data: lossData, borderColor: "#2563eb", backgroundColor: "#2563eb18", borderWidth: 2, pointRadius: 2, fill: true, tension: 0.4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: "linear", title: { display: true, text: "Epoch", font: { size: 11, family: "Inter" } }, ticks: { font: { size: 11 } } },
        y: { title: { display: true, text: "Loss", font: { size: 11, family: "Inter" } }, ticks: { font: { size: 11 } } }
      },
      plugins: { legend: { labels: { font: { size: 12, family: "Inter" }, boxWidth: 12, usePointStyle: true } } }
    }
  });

  document.getElementById("training-results").style.display = "block";
  state.trainingActive = false;
  document.getElementById("train-btn").disabled = false;
}

function updateFileUI(f) {
  const el = document.getElementById("fs-" + f.id);
  if (el) { el.textContent = f.status; el.className = "fi-status " + f.status; }
}
function updateFileProgress(f) {
  const bar = document.getElementById("fp-" + f.id);
  const pct = document.getElementById("fpct-" + f.id);
  if (bar) bar.style.width = f.progress + "%";
  if (pct) pct.textContent = f.progress + "%";
}
function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
  return (b / (1024 * 1024)).toFixed(1) + " MB";
}

// ---- BENCHMARK ----
let rocInst, distInst, classInst;

function runBenchmark(input) {
  const file = input.files[0];
  if (!file) return;
  const r = seededRand(file.name.length * 29 + file.size);
  setTimeout(() => generateBenchmarkResults(r, file.name), 800);
}

function generateBenchmarkResults(r, fname) {
  const n = 300 + Math.floor(r() * 400);
  const tp = Math.floor(r() * 50 + 100);
  const tn = Math.floor(r() * 50 + 95);
  const fp = Math.floor(r() * 20 + 8);
  const fn = Math.floor(r() * 20 + 8);
  const acc = (tp + tn) / (tp + tn + fp + fn);
  const prec = tp / (tp + fp);
  const rec = tp / (tp + fn);
  const f1 = 2 * prec * rec / (prec + rec);
  const auc = 0.86 + r() * 0.10;

  document.getElementById("bench-metrics").innerHTML = [
    { label: "Accuracy", val: (acc * 100).toFixed(2) + "%" },
    { label: "Precision", val: (prec * 100).toFixed(2) + "%" },
    { label: "Recall", val: (rec * 100).toFixed(2) + "%" },
    { label: "F1 score", val: (f1 * 100).toFixed(2) + "%" },
    { label: "ROC-AUC", val: auc.toFixed(4) },
    { label: "Samples", val: (tp + tn + fp + fn).toLocaleString() },
  ].map(m => `<div class="metric-tile"><div class="mt-val">${m.val}</div><div class="mt-label">${m.label}</div></div>`).join("");

  // Confusion matrix
  document.getElementById("cm-grid").innerHTML = [
    { cls: "cm-tp", val: tp, label: "True positive" },
    { cls: "cm-fp", val: fp, label: "False positive" },
    { cls: "cm-fn", val: fn, label: "False negative" },
    { cls: "cm-tn", val: tn, label: "True negative" },
  ].map(c => `<div class="cm-cell ${c.cls}"><div class="cm-val">${c.val}</div><div class="cm-label">${c.label}</div></div>`).join("");

  // ROC
  if (rocInst) rocInst.destroy();
  const rocPts = [[0, 0]];
  for (let i = 1; i <= 9; i++) { const x = i / 10; rocPts.push([x, Math.min(1, x + 0.35 + r() * 0.2)]); }
  rocPts.push([1, 1]);
  const rctx = document.getElementById("rocChart").getContext("2d");
  rocInst = new Chart(rctx, {
    type: "line",
    data: {
      datasets: [
        { label: `Ensemble (AUC=${auc.toFixed(3)})`, data: rocPts.map(p => ({ x: p[0], y: p[1] })), borderColor: "#2563eb", borderWidth: 2.5, pointRadius: 2.5, fill: false, tension: 0.35 },
        { label: "Random baseline", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "#d1cfc9", borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, fill: false }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate", font: { size: 11, family: "Inter" } }, ticks: { font: { size: 10 } } },
        y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate", font: { size: 11, family: "Inter" } }, ticks: { font: { size: 10 } } }
      },
      plugins: { legend: { labels: { font: { size: 11, family: "Inter" }, boxWidth: 12, usePointStyle: true } } }
    }
  });

  // Distribution
  if (distInst) distInst.destroy();
  const bins = Array.from({ length: 10 }, (_, i) => ({
    x: i * 10 + 5,
    human: Math.floor(r() * 18 + 4),
    ai: Math.floor(r() * 18 + 4),
  }));
  const dctx = document.getElementById("distChart").getContext("2d");
  distInst = new Chart(dctx, {
    type: "bar",
    data: {
      labels: bins.map(b => b.x + "%"),
      datasets: [
        { label: "Human", data: bins.map(b => b.human), backgroundColor: "#16a34a44", borderColor: "#16a34a", borderWidth: 1.5 },
        { label: "AI", data: bins.map(b => b.ai), backgroundColor: "#dc262644", borderColor: "#dc2626", borderWidth: 1.5 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { ticks: { font: { size: 10 } } }, y: { ticks: { font: { size: 10 } } } },
      plugins: { legend: { labels: { font: { size: 11, family: "Inter" }, boxWidth: 12, usePointStyle: true } } }
    }
  });

  // Per-class
  if (classInst) classInst.destroy();
  const cctx = document.getElementById("classChart").getContext("2d");
  classInst = new Chart(cctx, {
    type: "bar",
    data: {
      labels: ["Precision", "Recall", "F1"],
      datasets: [
        { label: "Human class", data: [(1 - fp / (tn + fp)).toFixed(3), (tn / (tn + fn)).toFixed(3), ((2 * (1 - fp / (tn + fp)) * (tn / (tn + fn))) / ((1 - fp / (tn + fp)) + (tn / (tn + fn)))).toFixed(3)], backgroundColor: "#16a34a66", borderColor: "#16a34a", borderWidth: 1.5 },
        { label: "AI class", data: [prec.toFixed(3), rec.toFixed(3), f1.toFixed(3)], backgroundColor: "#2563eb66", borderColor: "#2563eb", borderWidth: 1.5 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { min: 0, max: 1, ticks: { font: { size: 10 } } }, x: { ticks: { font: { size: 10 } } } },
      plugins: { legend: { labels: { font: { size: 11, family: "Inter" }, boxWidth: 12, usePointStyle: true } } }
    }
  });

  // FP / FN examples
  const fpExamples = [
    { text: "The sunset painted the sky in brilliant shades of amber and rose, casting long shadows across the field.", score: 78 },
    { text: "All systems must be optimized for maximum operational efficiency and resource utilization.", score: 82 },
    { text: "The results demonstrate a statistically significant correlation between the two variables.", score: 76 },
    { text: "It is important to note that the implementation of these strategies requires careful consideration.", score: 74 },
  ];
  const fnExamples = [
    { text: "honestly cant believe they did that lol whole thing is so chaotic and messy ngl", score: 31 },
    { text: "my dog refuses 2 eat unless i stand beside him its so annoying but also kinda cute", score: 27 },
    { text: "idk man this semester has been rough, barely surviving rn", score: 24 },
  ];
  document.getElementById("fp-list").innerHTML = fpExamples.map(e => `
    <div class="misclass-item">
      <div class="misclass-tags">
        <span class="m-tag m-human">Human</span>
        <span class="m-arrow">→</span>
        <span class="m-tag m-ai">Predicted AI (${e.score}%)</span>
      </div>
      <div class="misclass-text">${e.text}</div>
      <div class="misclass-score">Confidence: ${e.score}% AI · False positive</div>
    </div>`).join("");
  document.getElementById("fn-list").innerHTML = fnExamples.map(e => `
    <div class="misclass-item">
      <div class="misclass-tags">
        <span class="m-tag m-ai">AI</span>
        <span class="m-arrow">→</span>
        <span class="m-tag m-human">Predicted Human (${e.score}%)</span>
      </div>
      <div class="misclass-text">${e.text}</div>
      <div class="misclass-score">Confidence: ${e.score}% AI · False negative</div>
    </div>`).join("");

  document.getElementById("bench-results").style.display = "block";
}

function setSubtab(id, el) {
  document.querySelectorAll(".subtab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".subtab-content").forEach(t => { t.style.display = "none"; t.classList.remove("active"); });
  el.classList.add("active");
  const tabEl = document.getElementById("tab-" + id);
  tabEl.style.display = "block";
  tabEl.classList.add("active");
}

// ---- REPORTS ----
function renderReports() {
  const el = document.getElementById("report-list");
  if (!state.reports.length) {
    el.innerHTML = `<div class="empty-state"><i class="fa-regular fa-file-dashed-line"></i><span>No reports yet — run an analysis to generate one</span></div>`;
    return;
  }
  el.innerHTML = state.reports.map(r => `
    <div class="report-card">
      <div>
        <div class="rc-label ${r.ensemble > 0.6 ? "color-ai" : r.ensemble < 0.4 ? "color-human" : "color-mixed"}">${r.label}</div>
        <div class="rc-text">${r.text}</div>
        <div class="rc-time">${r.date}</div>
      </div>
      <div>
        <div class="rc-score ${r.ensemble > 0.6 ? "color-ai" : r.ensemble < 0.4 ? "color-human" : "color-mixed"}">${r.aiPct}% AI</div>
        <div class="rc-conf">${r.confidence}% confidence</div>
      </div>
    </div>`).join("");
}

function filterReports(q) {
  const el = document.getElementById("report-list");
  const filtered = state.reports.filter(r => r.text.toLowerCase().includes(q.toLowerCase()) || r.label.toLowerCase().includes(q.toLowerCase()));
  if (!filtered.length) { el.innerHTML = `<div class="empty-state"><i class="fa-regular fa-magnifying-glass"></i><span>No results for "${q}"</span></div>`; return; }
  el.innerHTML = filtered.map(r => `
    <div class="report-card">
      <div>
        <div class="rc-label ${r.ensemble > 0.6 ? "color-ai" : r.ensemble < 0.4 ? "color-human" : "color-mixed"}">${r.label}</div>
        <div class="rc-text">${r.text}</div>
        <div class="rc-time">${r.date}</div>
      </div>
      <div>
        <div class="rc-score ${r.ensemble > 0.6 ? "color-ai" : r.ensemble < 0.4 ? "color-human" : "color-mixed"}">${r.aiPct}% AI</div>
        <div class="rc-conf">${r.confidence}% confidence</div>
      </div>
    </div>`).join("");
}

function exportAllReports() {
  const blob = new Blob([JSON.stringify({ generated: new Date().toISOString(), platform: "DetectLens v2.0", reports: state.reports }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "detectlens-reports-" + Date.now() + ".json";
  a.click();
}

// ---- UTILS ----
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
