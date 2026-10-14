"use strict";

const BACKEND = "http://localhost:8000";  // for the "Backend:" label only; the service worker performs the request

// ---------------------------------------------------------------------------
// Page extractor. Injected into the active tab and run in the page (isolated
// world). MUST be fully self-contained: no closures over popup.js variables.
// Returns { text, source, title, company, wordCount, needsAssist }.
// It NEVER returns the whole page: if it can't confidently isolate the job
// description it returns needsAssist=true so the popup asks the user.
// ---------------------------------------------------------------------------
function extractJobPosting() {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const txt = (e) => clean(e ? (e.innerText || e.textContent) : "");
  const wc = (s) => (s ? s.split(/\s+/).filter(Boolean).length : 0);
  const MIN_SITE = 12, MIN_WORDS = 35, MAX_WORDS = 6000, MAX_CHARS = 15000;
  const host = (location.hostname || "").replace(/^www\./, "");

  // Non-job nodes stripped from inside the chosen container before reading text.
  const NOISE = [
    "nav","header","footer","aside","script","style","noscript","form","button",
    '[role="navigation"]','[role="complementary"]','[role="banner"]','[role="contentinfo"]',
    '[aria-hidden="true"]',
    '[class*="similar"]','[class*="recommend"]','[class*="related"]','[class*="more-job"]',
    '[class*="people-also"]','[class*="people-you"]','[class*="also-viewed"]',
    '[class*="comment"]','[class*="advert"]','[class*="promo"]','[class*="carousel"]',
    '[class*="navbar"]','[class*="nav-"]','[class*="menu"]','[class*="suggest"]',
    '[class*="footer"]','[id*="similar"]','[id*="recommend"]','[id*="footer"]',
    '[class*="job-match"]','[class*="job-alert"]','[class*="reach-out"]','[class*="feed-shared"]',
    '[class*="entity-lockup"]','[class*="hiring-team"]','[class*="premium"]','[class*="share-box"]',
    '[class*="salary-insight"]','[class*="careers-"]','[class*="top-card-layout__job-insight"]',
    '[class*="jobs-similar"]','[class*="jobs-recommend"]'
  ];
  const denoise = (el) => {
    if (!el) return "";
    const c = el.cloneNode(true);
    for (const q of NOISE) { try { c.querySelectorAll(q).forEach((n) => n.remove()); } catch (e) {} }
    return clean(c.innerText || c.textContent);
  };
  const pick = (sels) => { for (const q of sels) { const t = txt(document.querySelector(q)); if (t) return t; } return ""; };

  const title = txt(document.querySelector("h1")) || clean(document.title).split(/[|\u2013\u2014\-]/)[0].trim();
  const company = pick(['.job-details-jobs-unified-top-card__company-name','[data-testid*="company"]',
                        '[class*="company-name"]','[data-company]','a[href*="/company/"]']);
  const loc = pick(['[class*="location"]','[data-testid*="location"]','[class*="workplace"]']);
  const salary = pick(['[class*="salary"]','[class*="compensation"]','[data-testid*="salary"]']);

  const pack = (desc, source, words) => {
    const fields = {};
    if (title) fields["Job title"] = title;
    if (company) fields["Company"] = company;
    if (loc) fields["Location"] = loc;
    if (salary) fields["Salary"] = salary;
    fields["Job description"] = desc;
    const parts = [];
    for (const k of ["Job title","Company","Location","Salary"]) if (fields[k]) parts.push(k + ": " + fields[k]);
    parts.push(desc);
    return { text: clean(parts.join(". ")).slice(0, MAX_CHARS), source, title: title, company: company,
             wordCount: words, needsAssist: false, fields };
  };

  // 1) A substantial user text selection is the strongest signal of intent.
  const sel = clean(window.getSelection ? window.getSelection().toString() : "");
  if (wc(sel) >= 20) return pack(sel, "your selection", wc(sel));

  // 2) Site-specific containers (the site tags THE job ad) -> trusted, low floor.
  const SITE = {
    linkedin: [".jobs-description__container",".jobs-description-content__text",".jobs-search__job-details--container",".jobs-details__main-content .jobs-box__html-content",".jobs-details__main-content","#job-details",".jobs-description__content",".jobs-box__html-content",".show-more-less-html__markup",".description__text"],
    indeed: ["#jobDescriptionText"],
    glassdoor: [".jobDescriptionContent",'[class*="JobDetails_jobDescription"]'],
    jobstreet: ['[data-automation="jobAdDetails"]','[data-automation="jobDescription"]'],
    seek: ['[data-automation="jobAdDetails"]'],
    monster: [".job-description",'[class*="descriptionStyles"]'],
    ziprecruiter: [".job_description",'[class*="jobDescriptionSection"]']
  };
  const labels = host.split(".");   // brand-label match: jobstreet.com.my -> "jobstreet"
  let siteSel = [];
  for (const brand in SITE) if (labels.includes(brand)) { siteSel = SITE[brand].slice(); break; }
  for (const q of siteSel) {
    const t = denoise(document.querySelector(q));
    if (wc(t) >= MIN_SITE && wc(t) <= MAX_WORDS) return pack(t, "detected on " + host, wc(t));
  }

  // 3) Semantic / accessibility generalizers -> higher floor to avoid noise.
  const semantic = ['[id*="job-details"]','[id*="jobDescription"]','[class*="jobDescription"]','[class*="job-description"]',
    '[class*="jobs-description"]','[aria-label*="job description" i]','article[class*="job"]',
    'main [class*="description"]','[role="main"] [class*="description"]'];
  for (const q of semantic) {
    const t = denoise(document.querySelector(q));
    if (wc(t) >= MIN_WORDS && wc(t) <= MAX_WORDS) return pack(t, "detected on " + host, wc(t));
  }

  // 4) Generic fallback: the largest description-like block within a sane size.
  const cands = document.querySelectorAll('article, main, [role="main"], [class*="description"], [class*="Description"], [id*="description"], section');
  let bestT = "", bestW = 0;
  cands.forEach((el) => { const t = denoise(el); const w = wc(t); if (w >= MIN_WORDS && w <= MAX_WORDS && w > bestW) { bestW = w; bestT = t; } });
  // On a KNOWN job site, a huge generic block means the site-specific selectors
  // went stale and we are about to grab navigation / similar jobs / promos.
  // Prefer asking the user over shipping unrelated page content to the model.
  const isKnownSite = siteSel.length > 0;
  const KNOWN_SITE_GENERIC_CAP = 1200;
  if (bestW >= MIN_WORDS && !(isKnownSite && bestW > KNOWN_SITE_GENERIC_CAP)) {
    return pack(bestT, "auto-detected", bestW);
  }

  // 5) Could not confidently isolate the description -> user assist (never dump body).
  return { text: "", source: "none", title: title, company: company, wordCount: 0, needsAssist: true, fields: {} };
}

// ---------------------------------------------------------------------------
// Score helper — mirrors the web app's reconstruction so the extension shows
// the SAME final value: baseline + Σ rounded contributions.
// ---------------------------------------------------------------------------
function round2(v) { return Math.round((v + 1e-12) * 100) / 100; }
function sbFinalPct(sb) {
  if (!sb || !sb.available || !Array.isArray(sb.flagged) || typeof sb.reference_pct !== "number") return null;
  const ref = round2(sb.reference_pct);
  const pos = round2(sb.flagged.filter((f) => f.points > 0).reduce((s, f) => s + round2(f.points), 0));
  const neg = round2(sb.flagged.filter((f) => f.points < 0).reduce((s, f) => s + round2(f.points), 0));
  return round2(ref + pos + neg);
}
function riskClass(tier) {
  const s = String(tier || "").toLowerCase();
  if (s.includes("high")) return "high";
  if (s.includes("medium") || s.includes("mid")) return "medium";
  return "low";
}
function riskLabel(tier) {
  const c = riskClass(tier);
  return c === "high" ? "HIGH RISK" : c === "medium" ? "MEDIUM RISK" : "LOW RISK";
}

// ---------------------------------------------------------------------------
// DOM helpers + state machine
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const PANELS = ["loading", "assist", "error", "result"];
function show(panel) {
  for (const p of PANELS) $(p).classList.toggle("hidden", p !== panel);
}
function hideAllPanels() { for (const p of PANELS) $(p).classList.add("hidden"); }

let activeTab = null;

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function init() {
  $("backend").textContent = BACKEND.replace(/^https?:\/\//, "");
  try {
    activeTab = await getActiveTab();
    $("page-title").textContent = activeTab?.title || "(untitled page)";
    try { $("page-host").textContent = new URL(activeTab.url).hostname; } catch { $("page-host").textContent = ""; }
  } catch {
    $("page-title").textContent = "(no active tab)";
  }
  // Show whatever the background worker already has for this tab: an analysis
  // may be running or already finished from a previous popup session.
  if (activeTab) {
    try {
      const key = "flstate_" + activeTab.id;
      const o = await chrome.storage.session.get(key);
      renderFromState(o[key]);
    } catch (e) { /* no stored state yet */ }
  }
}

async function runExtraction() {
  hideAllPanels();
  $("loading-msg").textContent = "Extracting the job posting…";
  show("loading");

  if (!activeTab || !/^https?:/.test(activeTab.url || "")) {
    return showError("Open a job posting web page first",
      "The extension can only read normal web pages (http/https), not browser or extension pages.");
  }

  let extracted;
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId: activeTab.id },
      func: extractJobPosting
    });
    extracted = results && results[0] && results[0].result;
  } catch (e) {
    return showAssist(); // injection blocked (e.g. restricted page) -> let user paste
  }

  if (!extracted || extracted.needsAssist || !extracted.text) {
    return showAssist();
  }
  startAnalysis(extracted.text, extracted.source, extracted.wordCount);
}

function jobKey(text) {
  let h = 5381;
  for (let i = 0; i < text.length; i++) h = (((h << 5) + h + text.charCodeAt(i)) >>> 0);
  return String(h) + "_" + text.length;
}

// Hand the request to the background service worker. The worker owns the fetch
// and the analysis state, so the analysis survives the popup closing.
function startAnalysis(text, source, wordCount) {
  if (!text || text.trim().length < 10) {
    return showError("Not enough text to analyse", "The job description text was too short.");
  }
  hideAllPanels();
  $("loading-msg").textContent = "Analysing with FraudLens…";
  show("loading");
  if (!activeTab) return;
  chrome.runtime.sendMessage({
    type: "START", tabId: activeTab.id,
    text: text, source: source, wordCount: wordCount, jobKey: jobKey(text)
  });
  // The result arrives via chrome.storage.session: the onChanged listener
  // renders it if the popup is still open, and init() renders it on next open.
}

// Render whatever state the background worker currently holds for this tab.
function renderFromState(st) {
  if (!st || st.status === "idle") { hideAllPanels(); return; }
  if (st.status === "analysing") {
    hideAllPanels();
    $("loading-msg").textContent = "Analysing with FraudLens…";
    show("loading");
    return;
  }
  if (st.status === "completed") { renderResult(st.data, st.source, st.wordCount); return; }
  if (st.status === "error") { showError(st.title || "Something went wrong", st.error || ""); return; }
  hideAllPanels();
}

function factorBarRow(title, points, max) {
  const row = document.createElement("div"); row.className = "factor-row";
  const top = document.createElement("div"); top.className = "factor";
  const name = document.createElement("span"); name.className = "factor-name"; name.textContent = title;
  const val = document.createElement("span"); val.className = "factor-val"; val.textContent = "+" + Number(points).toFixed(2) + "%";
  top.appendChild(name); top.appendChild(val);
  const bar = document.createElement("div"); bar.className = "bar";
  const fill = document.createElement("span");
  fill.style.width = Math.max(4, Math.round((points / (max || 1)) * 100)) + "%";
  bar.appendChild(fill); row.appendChild(top); row.appendChild(bar);
  return row;
}
function kvRow(name, valText, opts) {
  opts = opts || {};
  const row = document.createElement("div"); row.className = "factor";
  const n = document.createElement("span"); n.className = "factor-name"; n.textContent = name;
  const v = document.createElement("span"); v.className = "factor-val"; v.textContent = valText;
  if (opts.muted) { n.style.color = "#9aa4b4"; }
  if (opts.neg) { v.style.color = "#e5779b"; }
  if (opts.base) { v.style.color = "#cdd3dc"; }
  row.appendChild(n); row.appendChild(v);
  return row;
}

function renderResult(data, source, wordCount) {
  hideAllPanels();

  const sb = data.signal_breakdown || {};
  const ref = (typeof sb.reference_pct === "number") ? round2(sb.reference_pct) : null;
  const flagged = sb.flagged || [];
  const positives = flagged.filter((f) => f.points > 0).sort((a, b) => b.points - a.points);
  const negatives = flagged.filter((f) => f.points < 0);
  const sumPos = round2(positives.reduce((s, f) => s + round2(f.points), 0));
  const sumNeg = round2(negatives.reduce((s, f) => s + round2(f.points), 0));

  // Displayed final: same reconstruction the web app uses. Falls back to the raw
  // model probability only when the breakdown is unavailable.
  const finalPct = (ref !== null) ? round2(ref + sumPos + sumNeg)
                                  : round2((data.fraud_probability || 0) * 100);
  $("score-pct").textContent = finalPct.toFixed(2) + "%";

  const chip = $("risk-chip");
  chip.textContent = riskLabel(data.risk_tier);
  chip.className = "chip " + riskClass(data.risk_tier);

  // Score breakdown: baseline + contributing factors, reconstructing the final.
  const fWrap = $("factors-wrap"), fBox = $("factors");
  fBox.innerHTML = "";
  if (ref !== null || positives.length) {
    if (ref !== null) fBox.appendChild(kvRow("Baseline (reference)", ref.toFixed(2) + "%", { muted: true, base: true }));
    const topN = positives.slice(0, 5);
    const max = (topN[0] && topN[0].points) || 1;
    for (const f of topN) fBox.appendChild(factorBarRow(f.title, f.points, max));
    if (positives.length > 5) {
      const rest = round2(positives.slice(5).reduce((s, f) => s + round2(f.points), 0));
      const more = positives.length - 5;
      fBox.appendChild(kvRow("+" + more + " more factor" + (more > 1 ? "s" : ""), "+" + rest.toFixed(2) + "%", { muted: true }));
    }
    if (negatives.length) fBox.appendChild(kvRow("Reduces the score", sumNeg.toFixed(2) + "%", { neg: true }));
    if (ref !== null) {
      const recon = document.createElement("div");
      recon.style.cssText = "margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.09);font-size:12px;color:#9aa4b4;font-variant-numeric:tabular-nums;";
      const negTxt = negatives.length ? (" \u2212 " + Math.abs(sumNeg).toFixed(2) + "%") : "";
      recon.textContent = "Baseline " + ref.toFixed(2) + "% + factors +" + sumPos.toFixed(2) + "%" + negTxt + " = " + finalPct.toFixed(2) + "%";
      fBox.appendChild(recon);
    }
    fWrap.classList.remove("hidden");
  } else {
    fWrap.classList.add("hidden");
  }

  // Rule-based checks (independent of the model)
  const rWrap = $("rules-wrap"), rBox = $("rules");
  rBox.innerHTML = "";
  const pa = data.pattern_analysis || {};
  const warnings = pa.warning || [];
  if (warnings.length) {
    for (const w of warnings.slice(0, 5)) rBox.appendChild(ruleRow(w.title, false));
    rWrap.classList.remove("hidden");
  } else {
    rBox.appendChild(ruleRow("No known scam warning patterns detected", true));
    for (const g of (pa.genuine || []).slice(0, 2)) rBox.appendChild(ruleRow(g.title, true));
    rWrap.classList.remove("hidden");
  }

  $("src-note").textContent = "Analysed " + (wordCount || "") + " words · source: " + (source || "page");
  show("result");
}

function ruleRow(title, ok) {
  const row = document.createElement("div");
  row.className = "rule" + (ok ? " ok" : "");
  const dot = document.createElement("span"); dot.className = "rule-dot";
  const txt = document.createElement("span"); txt.textContent = title;
  row.appendChild(dot); row.appendChild(txt);
  return row;
}

function showAssist() {
  hideAllPanels();
  show("assist");
}
function showError(title, msg) {
  hideAllPanels();
  $("error-title").textContent = title;
  $("error-msg").textContent = msg || "";
  show("error");
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  init();
  $("check-btn").addEventListener("click", runExtraction);
  $("assist-btn").addEventListener("click", () => {
    const t = $("assist-text").value || "";
    if (t.trim().split(/\s+/).filter(Boolean).length < 10) {
      return showError("Not enough text", "Please paste more of the job description (at least a couple of sentences).");
    }
    startAnalysis(t, "pasted text", t.trim().split(/\s+/).filter(Boolean).length);
  });
  $("retry-btn").addEventListener("click", runExtraction);
  $("again-btn").addEventListener("click", runExtraction);

  // Live-update the popup if the worker's state changes while it is open.
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "session" || !activeTab) return;
    const key = "flstate_" + activeTab.id;
    if (changes[key]) renderFromState(changes[key].newValue);
  });
});