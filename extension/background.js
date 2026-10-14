"use strict";

// FraudLens MV3 service worker.
// Owns the /predict/text request and the per-tab analysis state, so the
// analysis survives the popup closing / tab switches / screen sleep. The popup
// only sends commands and reads state; it never holds the long-running fetch.

const BACKEND = "http://localhost:8000";
const PREDICT_URL = BACKEND + "/predict/text";
const KEY = (tabId) => "flstate_" + tabId;

async function getState(tabId) {
  const o = await chrome.storage.session.get(KEY(tabId));
  return o[KEY(tabId)];
}
async function setState(tabId, st) {
  await chrome.storage.session.set({ [KEY(tabId)]: st });
}
// A result is stale if a newer job (different jobKey) started for this tab, or
// the tab's state was reset/removed while the request was in flight.
async function isStale(tabId, jobKey) {
  const s = await getState(tabId);
  return !s || s.jobKey !== jobKey;
}

async function startAnalysis({ tabId, text, source, wordCount, jobKey }) {
  const cur = await getState(tabId);
  // Ignore a duplicate click for the exact same job that is already running.
  if (cur && cur.status === "analysing" && cur.jobKey === jobKey) return;

  await setState(tabId, {
    status: "analysing", jobKey, source, wordCount, tabId, ts: Date.now()
  });

  try {
    const res = await fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    if (await isStale(tabId, jobKey)) return; // a newer job replaced this one
    if (!res.ok) {
      await setState(tabId, {
        status: "error", jobKey, tabId, ts: Date.now(),
        title: "FraudLens could not analyse this",
        error: res.status === 422
          ? "The posting text was rejected as too short."
          : ("The backend returned HTTP " + res.status + ".")
      });
      return;
    }
    const data = await res.json();
    if (await isStale(tabId, jobKey)) return;
    await setState(tabId, {
      status: "completed", jobKey, source, wordCount, data, tabId, ts: Date.now()
    });
  } catch (e) {
    // Only a genuine network failure reaches here — NOT the popup closing.
    if (await isStale(tabId, jobKey)) return;
    await setState(tabId, {
      status: "error", jobKey, tabId, ts: Date.now(),
      title: "Couldn't reach the FraudLens backend",
      error: "Make sure the backend is running at " + BACKEND +
             " (uvicorn app.main:app --port 8000)."
    });
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || !msg.type) return false;
  if (msg.type === "START") { startAnalysis(msg); return false; }
  if (msg.type === "RESET") { setState(msg.tabId, { status: "idle" }); return false; }
  return false;
});

// Clean up state when a tab is closed so old results don't linger.
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.session.remove(KEY(tabId));
});
