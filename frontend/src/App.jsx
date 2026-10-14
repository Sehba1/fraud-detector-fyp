import { useState, useRef, useEffect } from "react";
import { analyseText, analyseImage } from "./api/client";
import "./App.css";

function ParticleBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animationId;
    const mouse = { x: -9999, y: -9999 };

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    resize();

    class Particle {
      constructor() {
        this.reset();
      }
      reset() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 1.5 + 0.4;
        this.vx = (Math.random() - 0.5) * 0.4;
        this.vy = (Math.random() - 0.5) * 0.4;
        this.opacity = Math.random() * 0.5 + 0.15;
        this.hue = Math.random() > 0.5 ? 240 : 265;
      }
      update() {
        this.x += this.vx;
        this.y += this.vy;
        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          this.x -= dx * 0.015;
          this.y -= dy * 0.015;
        }
        if (
          this.x < -10 || this.x > canvas.width + 10 ||
          this.y < -10 || this.y > canvas.height + 10
        ) {
          this.reset();
        }
      }
      draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${this.hue}, 70%, 70%, ${this.opacity})`;
        ctx.fill();
      }
    }

    const particles = Array.from({ length: 120 }, () => new Particle());

    function drawConnections() {
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(99, 102, 241, ${0.14 * (1 - dist / 100)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
    }

    function animate() {
      ctx.fillStyle = "#050510";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      particles.forEach((p) => {
        p.update();
        p.draw();
      });
      drawConnections();
      animationId = requestAnimationFrame(animate);
    }
    animate();

    function onMove(e) {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    }
    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMove);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
    };
  }, []);

  return <canvas ref={canvasRef} className="particle-bg" aria-hidden="true" />;
}

function getRiskClass(tier) {
  if (tier === "high") return "high";
  if (tier === "medium") return "medium";
  return "low";
}

function getVerdictLabel(label, tier) {
  if (label === "fraudulent" && tier === "high") return "MOST LIKELY A SCAM";
  if (label === "fraudulent" && tier === "medium") return "PROCEED WITH CAUTION";
  return "LIKELY GENUINE";
}

function getVerdictSub(label, tier) {
  if (label === "fraudulent" && tier === "high")
    return "This posting shows strong warning signs. Treat it with caution.";
  if (label === "fraudulent" && tier === "medium")
    return "Some warning signs were detected. Check carefully before applying.";
  return "No major red flags were found, but this is not a guarantee.";
}

function getVerdictIcon(label, tier) {
  if (label === "fraudulent" && tier === "high") return "ti-alert-triangle";
  if (label === "fraudulent" && tier === "medium") return "ti-alert-circle";
  return "ti-circle-check";
}

function getAdvice(label, tier) {
  if (label === "fraudulent" && tier === "high") {
    return {
      icon: "ti-ban",
      text: (
        <>
          <strong>Do not share your bank details, passport, or personal information.</strong>{" "}
          This posting shows several patterns commonly found in job scams.
        </>
      ),
    };
  }
  if (label === "fraudulent" && tier === "medium") {
    return {
      icon: "ti-search",
      text: (
        <>
          <strong>Research this company independently</strong> before sharing any
          personal details. Check their official website and online reviews.
        </>
      ),
    };
  }
  return {
    icon: "ti-shield-check",
    text: (
      <>
        This posting shows the characteristics of a legitimate advertisement.{" "}
        <strong>Still verify the company independently</strong>, because our AI
        is not perfect and cannot check everything.
      </>
    ),
  };
}

function getRiskBadgeLabel(tier) {
  if (tier === "high") return "HIGH RISK";
  if (tier === "medium") return "MEDIUM RISK";
  return "LOW RISK";
}

function displayPercent(p) {
  const pct = p * 100;
  if (pct >= 99.9) return "99.9";
  if (pct > 0 && pct < 0.1) return "<0.1";
  return pct.toFixed(1);
}

function strengthOf(weight, maxWeight) {
  const ratio = Math.abs(weight) / maxWeight;
  if (ratio > 0.66) return "high";
  if (ratio > 0.33) return "med";
  return "low";
}

// Tokens shorter than this, or made only of digits and punctuation, are
// hidden from the word list. They are usually OCR fragments or currency
// prefixes and showing them as "the reason" looks broken.
function isDisplayableToken(word) {
  const w = (word || "").trim();
  if (w.length < 3) return false;
  if (!/[a-zA-Z]{3,}/.test(w)) return false;
  return true;
}

const PATTERN_ICONS = {
  external_contact: "ti-phone",
  unrealistic_income: "ti-coins",
  pressure: "ti-alarm",
  low_requirements: "ti-clipboard-x",
  money_requests: "ti-credit-card",
  identity_requests: "ti-id",
  targeting: "ti-user-exclamation",
  company_profile: "ti-building",
  clear_role: "ti-list-check",
  benefits: "ti-heart-handshake",
  qualifications: "ti-certificate",
  formal_process: "ti-briefcase",
};

const LIME_REASON_ICONS = {
  income: "ti-coins",
  urgency: "ti-alarm",
  financial_details: "ti-credit-card",
  easy_work: "ti-home",
  recruitment: "ti-speakerphone",
  contact: "ti-phone",
  qualifications: "ti-certificate",
  role: "ti-list-check",
  employment_terms: "ti-heart-handshake",
  process: "ti-briefcase",
  company: "ti-building",
  other: "ti-puzzle",
};

function OcclusionSection({ occlusion, label, riskClass }) {
  if (!occlusion) return null;

  const warnTone = riskClass === "medium" ? "caution" : "warn";

  if (!occlusion.available) {
    return (
      <div className="result-section">
        <div className="section-title">
          <i className="ti ti-target-arrow" aria-hidden="true" />
          What contributed to this score
        </div>
        <p className="counter-empty">
          <i className="ti ti-info-circle" aria-hidden="true" />
          {occlusion.reason}
        </p>
      </div>
    );
  }

  const hasAny = occlusion.raises.length > 0 || occlusion.lowers.length > 0;

  return (
    <div className="result-section">
      <div className="section-title">
        <i className="ti ti-target-arrow" aria-hidden="true" />
        What contributed to this score
      </div>

      <div className={`reasons-summary ${riskClass}`}>
        <i className="ti ti-target-arrow" aria-hidden="true" />
        <p>
          The overall score is <strong>{occlusion.baseline_pct}%</strong>. Each
          sentence below was removed and the posting re-scored. The change is
          that sentence's contribution, measured in score points.
        </p>
      </div>

      {!hasAny && (
        <p className="counter-empty">
          <i className="ti ti-info-circle" aria-hidden="true" />
          No single sentence changed the score by a measurable amount. The
          decision rests on the posting as a whole rather than any one part.
        </p>
      )}

      {occlusion.raises.length > 0 && (
        <>
          <div className="occ-group-label warn">
            Sentences pushing the score up
          </div>
          {occlusion.raises.map((s, i) => (
            <div key={i} className="occ-row">
              <div className="occ-head">
                <span className={`occ-points ${warnTone}`}>
                  +{s.impact_points} pts
                </span>
                <span className="occ-without">
                  score without this: {s.score_without_pct}%
                </span>
              </div>
              <p className="occ-sentence">{s.sentence}</p>
              <div className="occ-track">
                <div
                  className={`occ-fill ${warnTone}`}
                  style={{ width: `${Math.min(s.share_pct, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </>
      )}

      {occlusion.lowers.length > 0 && (
        <>
          <div className="occ-group-label safe">
            Sentences pushing the score down
          </div>
          {occlusion.lowers.map((s, i) => (
            <div key={i} className="occ-row">
              <div className="occ-head">
                <span className="occ-points safe">
                  -{Math.abs(s.impact_points)} pts
                </span>
                <span className="occ-without">
                  score without this: {s.score_without_pct}%
                </span>
              </div>
              <p className="occ-sentence">{s.sentence}</p>
              <div className="occ-track">
                <div
                  className="occ-fill safe"
                  style={{ width: `${Math.min(s.share_pct, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </>
      )}

      <p className="layer-note">
        <i className="ti ti-info-circle" aria-hidden="true" />
        These figures are measured directly from the model: each sentence is
        removed and the model runs again on what remains. {occlusion.sentences_tested}{" "}
        sentences were tested.
      </p>
    </div>
  );
}

function ScoreBreakdownSection({ breakdown, riskClass }) {
  if (!breakdown || !breakdown.available) return null;
  if (!breakdown.steps || breakdown.steps.length === 0) return null;

  const tone = riskClass === "medium" ? "caution" : "warn";
  const floor = breakdown.floor_pct;
  const final = breakdown.final_pct;

  // Largest step, used to scale the bar widths.
  const maxStep = Math.max(
    ...breakdown.steps.map((s) => Math.abs(s.step_points)),
    0.1
  );

  return (
    <div className="result-section">
      <div className="section-title">
        <i className="ti ti-stairs-up" aria-hidden="true" />
        How we reached this score
      </div>

      <div className={`reasons-summary ${riskClass}`}>
        <i className="ti ti-target-arrow" aria-hidden="true" />
        <p>
          The model starts from a <strong>baseline suspicion</strong> of{" "}
          <strong>{floor}%</strong>, then each factor below adds to the score.
          Together they build up to the final <strong>{final}%</strong>. The
          steps are measured on the model and shown biggest first.
        </p>
      </div>

      <div className="wf-track">
        <div className="wf-row wf-baseline">
          <span className="wf-label">Baseline suspicion</span>
          <span className={`wf-cum ${tone}`}>{floor}%</span>
        </div>

        {breakdown.steps.map((s, i) => (
          <div key={i} className="wf-row">
            <div className="wf-step-head">
              <span className={`wf-points ${tone}`}>+{s.step_points} pts</span>
              <span className="wf-after">running score: {s.score_after_pct}%</span>
            </div>
            <p className="wf-sentence">{s.sentence}</p>
            <div className="wf-bar">
              <div
                className={`wf-fill ${tone}`}
                style={{ width: `${Math.max((Math.abs(s.step_points) / maxStep) * 100, 6)}%` }}
              />
            </div>
          </div>
        ))}

        <div className="wf-row wf-final">
          <span className="wf-label">Final score</span>
          <span className={`wf-cum-final ${riskClass}`}>{final}%</span>
        </div>
      </div>

      <p className="layer-note">
        <i className="ti ti-info-circle" aria-hidden="true" />
        Each step is how much that factor adds to the model's score, measured on
        the model's raw output. Factors are added biggest first, so a factor's
        exact points depend on that order. The baseline plus every step add up to
        the final score.
      </p>
    </div>
  );
}

function CollapsibleSection({ title, icon, children }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="result-section">
      <button
        className="collapse-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <i className={`ti ${icon}`} aria-hidden="true" />
        <span>{title}</span>
        <i
          className={`ti ${open ? "ti-chevron-up" : "ti-chevron-down"} collapse-caret`}
          aria-hidden="true"
        />
      </button>
      {open && <div className="collapse-body">{children}</div>}
    </div>
  );
}

function PatternCard({ category, tone }) {
  const icon = PATTERN_ICONS[category.id] || "ti-point";
  const pct = category.model_attribution_pct;
  const ignored = pct != null && pct === 0;

  return (
    <div className="reason-card">
      <div className="reason-head">
        <div className={`reason-icon ${tone}`}>
          <i className={`ti ${icon}`} aria-hidden="true" />
        </div>
        <span className="reason-title">{category.title}</span>
      </div>

      <div className="reason-chips">
        {category.detected.map((snippet) => (
          <span key={snippet} className={`word-chip ${tone}`}>
            {snippet}
          </span>
        ))}
      </div>

      {pct != null && (
        <div className={`attrib-line ${ignored ? "ignored" : tone}`}>
          <i
            className={`ti ${ignored ? "ti-alert-circle" : "ti-brain"}`}
            aria-hidden="true"
          />
          <span>
            The AI model attributed <strong>{pct}%</strong> of its word-level
            attention to these words.
          </span>
        </div>
      )}

      <p className="pattern-explanation">{category.explanation}</p>
    </div>
  );
}

function LimeReasonCard({ group, tone }) {
  const icon = LIME_REASON_ICONS[group.id] || "ti-point";
  const words = group.words.filter((w) => isDisplayableToken(w.word));
  if (words.length === 0) return null;

  return (
    <div className="reason-card">
      <div className="reason-head">
        <div className={`reason-icon ${tone}`}>
          <i className={`ti ${icon}`} aria-hidden="true" />
        </div>
        <span className="reason-title">{group.title}</span>
        <span className={`card-points ${tone}`}>{group.contribution_pct}%</span>
      </div>
      <div className="attrib-track">
        <div
          className={`attrib-fill ${tone}`}
          style={{ width: `${Math.min(group.contribution_pct, 100)}%` }}
        />
      </div>
      <div className="reason-chips">
        {words.map((w) => (
          <span
            key={w.word}
            className={`word-chip ${tone}`}
            title={`LIME weight: ${w.weight}`}
          >
            {w.word}
          </span>
        ))}
      </div>
    </div>
  );
}

function LimeReasonsSection({ reasons, label, riskClass, fidelity }) {
  const tone =
    label === "fraudulent"
      ? riskClass === "medium"
        ? "caution"
        : "warn"
      : "safe";

  const cards = reasons.groups
    .map((g) => <LimeReasonCard key={g.id} group={g} tone={tone} />)
    .filter(Boolean);

  return (
    <div className="result-section">
      <div className="section-title">
        <i className="ti ti-chart-pie" aria-hidden="true" />
        Word-level attention (supporting detail)
      </div>
      {cards.length > 0 ? (
        <div className="reason-grid">{cards}</div>
      ) : (
        <p className="counter-empty">
          <i className="ti ti-info-circle" aria-hidden="true" />
          No individual words carried enough weight to group.
        </p>
      )}
      <p className="layer-note">
        <i className="ti ti-info-circle" aria-hidden="true" />
        These percentages divide up the model's word-level attention. They are
        shares of the explanation, not parts of the score above.
        {fidelity != null && (
          <> Local explanation fidelity: {fidelity}.</>
        )}
      </p>
    </div>
  );
}

function PatternAnalysisSection({ analysis, label, riskClass }) {
  const warnTone = riskClass === "medium" ? "caution" : "warn";
  const hasWarnings = analysis.warning.length > 0;
  const hasGenuine = analysis.genuine.length > 0;
  const warnBannerClass = riskClass === "low" ? "medium" : riskClass;

  const ignoredWarnings = analysis.warning.filter(
    (c) => c.model_attribution_pct === 0
  );

  return (
    <>
      <div className="result-section">
        <div className="section-title">
          <i className="ti ti-alert-triangle" aria-hidden="true" />
          Warning signs found in this posting
        </div>
        {hasWarnings ? (
          <>
            <div className={`reasons-summary ${warnBannerClass}`}>
              <i className="ti ti-alert-circle" aria-hidden="true" />
              <p>{analysis.warning_summary}</p>
            </div>
            <div className="reason-grid">
              {analysis.warning.map((cat) => (
                <PatternCard key={cat.id} category={cat} tone={warnTone} />
              ))}
            </div>
            {ignoredWarnings.length > 0 && (
              <p className="layer-note">
                <i className="ti ti-info-circle" aria-hidden="true" />
                {ignoredWarnings.length === 1
                  ? "One of these warning signs carried no weight in the model's decision."
                  : `${ignoredWarnings.length} of these warning signs carried no weight in the model's decision.`}{" "}
                They are shown because they are documented fraud indicators,
                even though the model did not respond to them.
              </p>
            )}
          </>
        ) : (
          <>
            <p className="counter-empty">
              <i className="ti ti-circle-check" aria-hidden="true" />
              {analysis.warning_summary}
            </p>
            {label === "fraudulent" && (
              <p className="layer-note">
                <i className="ti ti-info-circle" aria-hidden="true" />
                The model classified this posting as fraudulent from its overall
                language, even though none of the known scam patterns were
                matched.
              </p>
            )}
          </>
        )}
      </div>

      <div className="result-section">
        <div className="section-title">
          <i className="ti ti-shield-check" aria-hidden="true" />
          Signals that look genuine
        </div>
        {hasGenuine ? (
          <>
            <div className="reasons-summary low">
              <i className="ti ti-circle-check" aria-hidden="true" />
              <p>{analysis.genuine_summary}</p>
            </div>
            <div className="reason-grid">
              {analysis.genuine.map((cat) => (
                <PatternCard key={cat.id} category={cat} tone="safe" />
              ))}
            </div>
          </>
        ) : (
          <p className="counter-empty">
            <i className="ti ti-info-circle" aria-hidden="true" />
            {analysis.genuine_summary}
          </p>
        )}
      </div>
    </>
  );
}

function HighlightedText({ text, explanation, riskClass }) {
  if (!explanation || explanation.length === 0) {
    return <div className="posting-text-box">{text}</div>;
  }

  const wordMap = {};
  explanation.forEach(({ word, weight }) => {
    if (isDisplayableToken(word)) {
      wordMap[word.toLowerCase()] = weight;
    }
  });

  const maxWeight = Math.max(
    ...explanation.map((e) => Math.abs(e.weight)),
    0.0001
  );

  const tokens = text.split(/(\s+)/);
  return (
    <div className="posting-text-box">
      {tokens.map((token, i) => {
        const clean = token.toLowerCase().replace(/[^a-z]/g, "");
        const weight = wordMap[clean];
        if (weight !== undefined) {
          const isWarning = weight > 0;
          const hlClass = isWarning
            ? riskClass === "medium"
              ? "hl-caution"
              : "hl-warn"
            : "hl-safe";
          const kind = isWarning ? "Higher model attribution" : "Leaned toward genuine";
          return (
            <mark
              key={i}
              className={hlClass}
              title={`${kind}, strength: ${strengthOf(weight, maxWeight)}`}
            >
              {token}
            </mark>
          );
        }
        return <span key={i}>{token}</span>;
      })}
    </div>
  );
}

function ResultsPanel({ result, inputText }) {
  const riskClass = getRiskClass(result.risk_tier);

  const moneyRequested = result.pattern_analysis?.warning?.some(
    (c) => c.id === "money_requests"
  );
  const advice =
    result.label === "genuine" && moneyRequested
      ? {
          icon: "ti-alert-triangle",
          text: (
            <>
              <strong>This posting asks you to pay money.</strong> Legitimate
              employers never charge a fee for a job or internship. Treat this
              offer with extreme caution, no matter what the result above says.
            </>
          ),
        }
      : getAdvice(result.label, result.risk_tier);

  const hasOcr = result.input_type === "image" && result.ocr_text;
  const displayText = hasOcr ? result.ocr_text : inputText;

  // One reconciled Scam probability at the top. Computed as baseline + Σφ
  // using the same rounded values as the Score Breakdown tab, so the number
  // shown here matches the tab exactly. Underlying Shapley values are not
  // modified; this is presentation-only.
  const reconciledFinalPct = _sbFinalPct(result.signal_breakdown);

  return (
    <div className="results-panel">
      <div className={`result-card ${riskClass}`}>
        <div className="verdict-row">
          <div className="v-orb">
            <i
              className={`ti ${getVerdictIcon(result.label, result.risk_tier)}`}
              aria-hidden="true"
            />
          </div>
          <div className="verdict-main">
            <div className="verdict-label">
              {getVerdictLabel(result.label, result.risk_tier)}
            </div>
            <div className="verdict-sub">
              {getVerdictSub(result.label, result.risk_tier)}
            </div>
            <div className="advice-box">
              <i className={`ti ${advice.icon}`} aria-hidden="true" />
              <p>{advice.text}</p>
            </div>
          </div>
          <div className="verdict-metrics">
            <div>
              <span className="metric-val">
                {reconciledFinalPct != null
                  ? `${reconciledFinalPct.toFixed(2)}%`
                  : `${displayPercent(result.fraud_probability)}%`}
              </span>
              <span className="metric-lbl">Scam probability</span>
            </div>
            <div>
              <span className="metric-val">
                <span className={`risk-badge ${riskClass}`}>
                  {getRiskBadgeLabel(result.risk_tier)}
                </span>
              </span>
              <span className="metric-lbl">Risk level</span>
            </div>
          </div>
        </div>

        <div className="unified-flow">
          <SignalBreakdownSection
            breakdown={result.signal_breakdown}
            label={result.label}
            original={displayText}
            patternAnalysis={result.pattern_analysis}
            riskClass={riskClass}
          />

          {hasOcr && (
            <div className="result-section">
              <div className="section-title">
                <i className="ti ti-camera" aria-hidden="true" />
                Text we read from your picture
              </div>
              <div className="ocr-block">
                <p className="ocr-text">{result.ocr_text}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- AI Model Analysis: tabbed score breakdown + explanations (dependency-free SVG) ---
function _polar(cx, cy, r, deg) {
  const a = ((deg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}
function _donutSlice(cx, cy, rOut, rIn, startDeg, endDeg) {
  const [ox1, oy1] = _polar(cx, cy, rOut, startDeg);
  const [ox2, oy2] = _polar(cx, cy, rOut, endDeg);
  const [ix2, iy2] = _polar(cx, cy, rIn, endDeg);
  const [ix1, iy1] = _polar(cx, cy, rIn, startDeg);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${ox1} ${oy1} A ${rOut} ${rOut} 0 ${large} 1 ${ox2} ${oy2} L ${ix2} ${iy2} A ${rIn} ${rIn} 0 ${large} 0 ${ix1} ${iy1} Z`;
}
const PIE_COLORS = ["#e5484d", "#f5a623", "#e0679b", "#7c5cff", "#2ea3f2", "#12a594", "#b7742f", "#d6409f"];
const BASELINE_COLOR = "#8a94a6";

const OTHER_FACTOR_ID = "other_posting_content";

const WHY_TEXT = {
  unrealistic_income_commission: "The posting emphasises high or fast earnings or commission relative to the limited job detail provided. Unusually high income claims are a pattern that can appear in fraudulent recruitment adverts.",
  remote_data_entry: "The posting leans on remote, work-from-home or data-entry framing. This wording appears in many legitimate jobs but is also heavily represented in known recruitment scams, so the model weighs it.",
  urgency_pressure: "The posting uses urgency or pressure, such as an immediate start or limited positions. Pressure to act quickly is a common tactic in fraudulent adverts.",
  low_no_requirements: "The posting asks for little or no experience or qualifications. Very low entry requirements can appear in scams that target a wide pool of applicants.",
  targeting_language: "The posting targets applicants by attributes such as gender or age. This kind of targeting is unusual for professional employers and appears in some fraudulent local adverts.",
  contact_recruitment: "The posting directs applicants to contact or apply through channels such as messaging apps or a direct phone number rather than an official recruitment process.",
  fees_payment_request: "The posting references fees, deposits or payments. Requests for money from applicants are a strong indicator of recruitment fraud.",
  personal_bank_information: "The posting references bank or personal identity details. Requests for sensitive personal or financial information early in hiring can indicate fraud.",
  other_posting_content: "This is a catch-all group covering posting text that did not match any predefined category. Its contribution measures how much the model relied on that residual content when making its decision, spread across the general language of the posting rather than concentrated in one phrase.",
};
function whyText(id) {
  return WHY_TEXT[id] || "This factor contributed to the model's score based on the detected evidence shown below.";
}
function round2(v) { return Math.round((v + Number.EPSILON) * 100) / 100; }

// Evidence selection is grounded in the category-defining keywords that ACTUALLY
// appear in the factor's grouped text (backend `keywords`). A sentence is chosen for
// a factor only if it contains such a term, so a sentence is never matched to a factor
// it does not semantically support. Generic; no per-posting phrases are hard-coded.
//
// Confidence gate: evidence is shown only when the model's own Shapley contribution
// to this group is at least EVIDENCE_MIN percentage points. Below that the group's
// textual "presence" is incidental — the FACTORS keyword sets are polarity-blind
// (they can't distinguish "no diploma required" from "Bachelor's degree required")
// so at small attributions any surfaced sentence is likely a false positive. The
// gate uses the ONE reliable signal — how much the model actually relied on this
// group — and defaults to "No specific evidence identified." otherwise. A safe
// non-answer is better than a misleading explanation.
const EVIDENCE_MIN = 1.0;
function _meaningfulKeywords(factor) {
  // require reasonably specific terms; drops weak/ambiguous short tokens like "fast", "now", "only", "data"
  return Array.from(new Set((factor.keywords || []).map((w) => w.toLowerCase()).filter((w) => w.length >= 5)));
}
function _tokenMatches(tok, kw) {
  if (tok === kw) return true;
  if (kw.length >= 5 && tok.startsWith(kw)) return true;
  if (tok.length >= 5 && kw.startsWith(tok)) return true;
  return false;
}
// Truthful evidence selection.
//
// Two independent sources are consulted, both derived from the model's own
// behaviour on the posting; NEITHER fabricates or hard-codes evidence.
//
// 1. Category-keyword sentence match (primary). Uses the backend `keywords`
//    field — the category-defining terms that actually appear in this factor's
//    grouped text — to find a sentence in the ORIGINAL posting that contains
//    at least one such term. This gives a clean sentence tied semantically to
//    the factor's name (e.g. "female" for Targeting, "whatsapp" for Contact).
//
// 2. Backend `words` fallback. The backend records, per factor, the unit texts
//    whose per-unit sensitivity |eff| > 0.002 — i.e. the units that measurably
//    changed the model's probability when replaced with genuine-background
//    content. When (1) yields nothing this fallback surfaces those model-picked
//    units, so:
//       - "Other posting content" (a catch-all with no category keywords by
//         design) can still show honest, model-derived evidence rather than
//         "No specific evidence identified."
//       - named factors whose only appearing keyword was short enough to be
//         dropped by the meaningful-keyword filter (e.g. `home` in Remote /
//         data-entry) still surface evidence.
//
// The EVIDENCE_MIN gate applies uniformly, so tiny attributions never surface
// any evidence and cannot be misread as detected signals.
function pickEvidence(factor, original) {
  if (Math.abs((factor.points != null ? factor.points : 0)) < EVIDENCE_MIN) return null;

  // 1. Keyword-based sentence match against the original posting.
  const keywords = _meaningfulKeywords(factor);
  if (original && keywords.length) {
    const sentences = original.split(/(?<=[.!?])\s+|\n+/).map((s) => s.trim())
      .filter((s) => s.split(/\s+/).filter(Boolean).length >= 5);
    let best = null, bestScore = 0;
    for (const s of sentences) {
      const toks = new Set(s.toLowerCase().split(/[^a-z]+/).filter(Boolean));
      let hits = 0;
      for (const kw of keywords) {
        for (const t of toks) { if (_tokenMatches(t, kw)) { hits += 1; break; } }
      }
      const score = hits + Math.min(s.length, 200) / 100000;
      if (hits > 0 && score > bestScore) { bestScore = score; best = s; }
    }
    if (best) return best;
  }

  // 2. Fallback: use the backend-supplied top unit text for this factor.
  //    Backend `words` are unit raw texts filtered by |eff| > 0.002, so they
  //    are already model-derived evidence for this factor. This covers
  //    "Other posting content" (which has no keywords) and named factors
  //    whose only appearing keyword was too short to survive the meaningful
  //    filter (e.g. "home" for Remote / data-entry).
  if (Array.isArray(factor.words) && factor.words.length > 0) {
    // Prefer the top-|eff| unit text as-is; it comes from the original text
    // (unit.raw), truncated to 80 chars by the backend.
    return factor.words[0];
  }
  return null;
}

// Reconciled reconstructed final = baseline + Σφ, using the SAME rounded
// values as the Score Breakdown tab so the number shown at the top of the
// results card and the number shown inside the tab are guaranteed to match.
// Underlying Shapley values are unchanged; this is a presentation-only helper.
function _sbFinalPct(breakdown) {
  if (!breakdown || !breakdown.available) return null;
  const flagged = breakdown.flagged || [];
  const reference = round2(typeof breakdown.reference_pct === "number" ? breakdown.reference_pct : 0);
  const pts = flagged.map((f) => round2(f.points));
  const sumPos = round2(pts.filter((v) => v > 0).reduce((s, v) => s + v, 0));
  const sumNeg = round2(pts.filter((v) => v < 0).reduce((s, v) => s + v, 0));
  return round2(reference + sumPos + sumNeg);
}

function SignalBreakdownSection({ breakdown, label, original, patternAnalysis, riskClass }) {
  const [tab, setTab] = useState("breakdown");
  const [hovered, setHovered] = useState(null);
  if (!breakdown) return null;
  const flagged = (breakdown.flagged || []).slice();
  const hasFactors = breakdown.available && flagged.length > 0;
  const isFraud = label === "fraudulent";
  const hasPatterns = !!(patternAnalysis && (
    ((patternAnalysis.warning || []).length + (patternAnalysis.genuine || []).length) > 0
  ));
  if (!hasFactors && !isFraud && !hasPatterns) return null;

  const reference = round2(typeof breakdown.reference_pct === "number" ? breakdown.reference_pct : 0);

  // ONE source of truth: the rounded flagged factors drive both tabs and the chart.
  const factors = flagged.map((f) => ({ ...f, pts: round2(f.points) }));
  const positives = factors.filter((f) => f.pts > 0).sort((a, b) => b.pts - a.pts);
  const negatives = factors.filter((f) => f.pts < 0).sort((a, b) => a.pts - b.pts);
  const sumPos = round2(positives.reduce((s, f) => s + f.pts, 0));
  const sumNeg = round2(negatives.reduce((s, f) => s + f.pts, 0));
  const finalScore = round2(reference + sumPos + sumNeg);
  const fmt = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;

  const pieItems = [{ id: "__baseline__", title: "Baseline (reference)", pts: reference, color: BASELINE_COLOR }]
    .concat(positives.map((f, i) => ({ ...f, color: PIE_COLORS[i % PIE_COLORS.length] })));
  const pieTotal = pieItems.reduce((s, it) => s + it.pts, 0) || 1;
  let acc = 0;
  const slices = pieItems.map((it) => {
    const start = (acc / pieTotal) * 360;
    acc += it.pts;
    const end = (acc / pieTotal) * 360;
    const mid = (start + end) / 2;
    const isHover = hovered === it.id;
    const [dx, dy] = _polar(0, 0, isHover ? 6 : 0, mid);
    return { ...it, d: _donutSlice(100 + dx, 96 + dy, 82, 50, start, end) };
  });
  const centre = hovered ? pieItems.find((p) => p.id === hovered) : null;
  const orderedFactors = factors.slice().sort((a, b) => b.pts - a.pts);

  return (
    <div className="result-section breakdown-section">
      <div className="model-section-header">AI Model Analysis</div>
      <div className="xai-tabs" role="tablist">
        <button type="button" role="tab" className={`xai-tab ${tab === "breakdown" ? "active" : ""}`} onClick={() => setTab("breakdown")}>Score Breakdown</button>
        <button type="button" role="tab" className={`xai-tab ${tab === "why" ? "active" : ""}`} onClick={() => setTab("why")}>Why This Result</button>
        <button type="button" role="tab" className={`xai-tab ${tab === "rules" ? "active" : ""}`} onClick={() => setTab("rules")}>Rule-Based Checks</button>
      </div>

      {tab === "breakdown" && (
        <div className="xai-panel">
          {!hasFactors ? (
            <div className="weak-explanation-note">
              Limited explanation: the model detected patterns associated with this prediction,
              but the available attribution provides limited contextual evidence.
            </div>
          ) : (<>
          <div className="breakdown-grid">
            <svg className="score-donut" viewBox="0 0 200 200" role="img" aria-label="Score composition donut chart">
              <defs>
                <filter id="donutShadow" x="-25%" y="-25%" width="150%" height="150%">
                  <feDropShadow dx="0" dy="5" stdDeviation="4" floodColor="#000000" floodOpacity="0.45" />
                </filter>
              </defs>
              <ellipse cx="100" cy="106" rx="82" ry="80" fill="#000000" opacity="0.35" />
              <g filter="url(#donutShadow)">
                {slices.map((s) => (
                  <path
                    key={s.id}
                    d={s.d}
                    fill={s.color}
                    stroke="#0b0f17"
                    strokeWidth="1.5"
                    opacity={hovered && hovered !== s.id ? 0.4 : 1}
                    onMouseEnter={() => setHovered(s.id)}
                    onMouseLeave={() => setHovered(null)}
                    style={{ cursor: "pointer", transition: "opacity 0.15s" }}
                  />
                ))}
              </g>
              <text x="100" y="90" textAnchor="middle" className="donut-centre-top">{centre ? centre.title.slice(0, 24) : "Final score"}</text>
              <text x="100" y="112" textAnchor="middle" className="donut-centre-val">{centre ? (centre.id === "__baseline__" ? `${centre.pts.toFixed(2)}%` : fmt(centre.pts)) : `${finalScore.toFixed(2)}%`}</text>
            </svg>
            <div className="pie-legend">
              {pieItems.map((it) => (
                <div
                  className={`pie-legend-row ${hovered === it.id ? "hl" : ""}`}
                  key={it.id}
                  onMouseEnter={() => setHovered(it.id)}
                  onMouseLeave={() => setHovered(null)}
                >
                  <span className="legend-dot" style={{ background: it.color }} />
                  <span className="legend-name">{it.title}</span>
                  <span className="legend-val">{it.id === "__baseline__" ? `${it.pts.toFixed(2)}%` : fmt(it.pts)}</span>
                </div>
              ))}
            </div>
          </div>

          {negatives.length > 0 && (
            <div className="neg-block">
              <div className="neg-title">Factors that REDUCE the score (shown separately, not in the donut)</div>
              {negatives.map((f) => (
                <div className="neg-row" key={f.id}>
                  <span className="legend-name">{f.title}</span>
                  <span className="neg-val">{f.pts.toFixed(2)}%</span>
                </div>
              ))}
            </div>
          )}

          <div className="reconstruction">
            <span className="recon-part">Baseline {reference.toFixed(2)}%</span>
            <span className="recon-op">+</span>
            <span className="recon-part">positive contributions {sumPos.toFixed(2)}%</span>
            {negatives.length > 0 && (
              <>
                <span className="recon-op">&minus;</span>
                <span className="recon-part">reductions {Math.abs(sumNeg).toFixed(2)}%</span>
              </>
            )}
            <span className="recon-op">=</span>
            <span className="recon-final">{finalScore.toFixed(2)}%</span>
          </div>
          <div className="recon-note">
            Baseline plus every displayed factor contribution reconstructs the final scam probability shown at the top of this page.
          </div>
          </>)}
        </div>
      )}

      {tab === "why" && (
        <div className="xai-panel">
          {!hasFactors ? (
            <p className="counter-empty">
              <i className="ti ti-info-circle" aria-hidden="true" />
              No factor-level attribution was produced for this posting.
            </p>
          ) : (<>
          <p className="why-intro">
            Each item is a predefined content group the model&rsquo;s scam probability was measured against, most influential first.
            The percentages in Score Breakdown are genuine model attribution; the group names are human-defined labels, not categories the model was trained to recognise.
          </p>
          {orderedFactors.map((f) => {
            const ev = pickEvidence(f, original);
            const isOther = f.id === OTHER_FACTOR_ID;
            const evLabel = isOther ? "Representative model-weighted text:" : "Detected evidence:";
            return (
              <div className="why-card" key={f.id}>
                <div className="why-head">
                  <span className="why-name">{f.title}</span>
                </div>
                {ev ? (
                  <div className="why-evidence"><span className="why-ev-label">{evLabel}</span> &ldquo;{ev}&rdquo;</div>
                ) : (
                  <div className="why-evidence muted">No specific evidence identified.</div>
                )}
                <div className="why-explain"><span className="why-ev-label">Why this matters:</span> {whyText(f.id)}</div>
              </div>
            );
          })}
          </>)}
        </div>
      )}

      {tab === "rules" && (
        <div className="xai-panel">
          {patternAnalysis ? (
            <PatternAnalysisSection
              analysis={patternAnalysis}
              label={label}
              riskClass={riskClass}
            />
          ) : (
            <p className="counter-empty">
              <i className="ti ti-info-circle" aria-hidden="true" />
              No rule-based checks are available for this posting.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
export default function App() {
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [inputSnapshot, setInputSnapshot] = useState("");
  const [scanStep, setScanStep] = useState(0);
  const fileRef = useRef(null);

  useEffect(() => {
    if (!loading) {
      setScanStep(0);
      return;
    }
    const t1 = setTimeout(() => setScanStep(1), 3000);
    const t2 = setTimeout(() => setScanStep(2), 20000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [loading]);

  function reset() {
    setText("");
    setFile(null);
    setFileName("");
    setResult(null);
    setError(null);
    setInputSnapshot("");
    if (fileRef.current) fileRef.current.value = "";
  }

  function switchMode(m) {
    setMode(m);
    reset();
  }

  async function handleSubmit() {
    setError(null);
    setResult(null);

    if (mode === "text" && text.trim().length < 10) {
      setError("Please paste a job posting with at least a few sentences.");
      return;
    }
    if (mode === "image" && !file) {
      setError("Please upload a picture of the job posting first.");
      return;
    }

    setLoading(true);
    setInputSnapshot(text);

    try {
      const data =
        mode === "text" ? await analyseText(text) : await analyseImage(file);
      setResult(data);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Something went wrong. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  function handleFile(e) {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      setFileName(f.name);
      setResult(null);
      setError(null);
    }
  }

  const stepClass = (idx) => {
    if (scanStep > idx) return "done";
    if (scanStep === idx) return "active";
    return "wait";
  };

  return (
    <>
      <ParticleBackground />
      <div className="app-content">
        <header className="header">
          <div className="logo-mark">
            <i className="ti ti-zoom-scan" aria-hidden="true" />
          </div>
          <span className="logo-name">FraudLens</span>
          <span className="logo-tagline">
            AI-powered job scam detector · APU Final Year Project
          </span>
        </header>

        <main className="main">
          <section className="hero">
            <div className="hero-badge">
              <div className="badge-dot" />
              AI-powered · Real-time · Free to use
            </div>
            <h1 className="hero-title">
              <span className="hero-line1">Is this job posting real?</span>
              <span className="hero-line2">Let AI decide in seconds.</span>
            </h1>
            <p className="hero-sub">
              Paste a job ad or upload a picture. Our AI reads every word,
              spots scam patterns, and explains exactly what raised the alarm
              in plain English.
            </p>
          </section>

          <div className="card-3d">
            <div className="mode-tabs">
              <button
                className={`tab-btn ${mode === "text" ? "active" : ""}`}
                onClick={() => switchMode("text")}
              >
                <i className="ti ti-text-size" aria-hidden="true" />
                Paste text
              </button>
              <button
                className={`tab-btn ${mode === "image" ? "active" : ""}`}
                onClick={() => switchMode("image")}
              >
                <i className="ti ti-photo-up" aria-hidden="true" />
                Upload picture
              </button>
            </div>

            <div className="input-area">
              {mode === "text" ? (
                <textarea
                  className="text-input"
                  placeholder="Paste the full job posting here. Include the title, description, salary, and requirements if you have them."
                  value={text}
                  onChange={(e) => {
                    setText(e.target.value);
                    setResult(null);
                    setError(null);
                  }}
                  rows={8}
                />
              ) : (
                <div
                  className="upload-zone"
                  onClick={() => fileRef.current?.click()}
                >
                  <input
                    ref={fileRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp,image/bmp"
                    onChange={handleFile}
                    style={{ display: "none" }}
                  />
                  {fileName ? (
                    <div className="upload-selected">
                      <div className="upload-icon-wrap">
                        <i className="ti ti-photo-check" aria-hidden="true" />
                      </div>
                      <span className="upload-filename">{fileName}</span>
                      <span className="upload-change">
                        Click to choose a different picture
                      </span>
                    </div>
                  ) : (
                    <>
                      <div className="upload-icon-wrap">
                        <i className="ti ti-photo-up" aria-hidden="true" />
                      </div>
                      <span className="upload-main">
                        Click to upload a picture
                      </span>
                      <span className="upload-sub">
                        JPEG, PNG, WebP or BMP. Screenshots of job ads work
                        great.
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>

            {error && (
              <div className="error-wrap">
                <div className="error-box">
                  <i className="ti ti-alert-circle" aria-hidden="true" />
                  {error}
                </div>
              </div>
            )}

            <div className="action-row">
              <span className="action-hint">
                <i className="ti ti-clock" aria-hidden="true" />
                This may take a moment
              </span>
              {(result || error) && (
                <button className="btn-clear" onClick={reset}>
                  <i className="ti ti-refresh" aria-hidden="true" />
                  Check another
                </button>
              )}
              <button
                className="btn-scan"
                onClick={handleSubmit}
                disabled={loading}
              >
                {loading ? (
                  <span className="loading-inner">
                    <span className="spinner" />
                    CHECKING
                  </span>
                ) : (
                  <>
                    <i className="ti ti-scan" aria-hidden="true" />
                    CHECK THIS POSTING
                  </>
                )}
              </button>
            </div>
          </div>

          {loading && (
            <div className="card-3d">
              <div className="scan-wrap">
                <div className="scan-orb">
                  <div className="orb-ring1" />
                  <div className="orb-ring2" />
                  <div className="orb-ring3" />
                  <div className="orb-core">
                    <i className="ti ti-scan" aria-hidden="true" />
                  </div>
                </div>
                <div className="scan-title">Checking your posting</div>
                <div className="scan-sub">
                  Reading every word and looking for warning signs
                </div>
                <div className="scan-steps">
                  <div className="scan-step">
                    <div className={`step-dot ${stepClass(0)}`} />
                    <span className={`step-text ${stepClass(0)}`}>
                      Cleaning and preparing the text
                    </span>
                  </div>
                  <div className="scan-step">
                    <div className={`step-dot ${stepClass(1)}`} />
                    <span className={`step-text ${stepClass(1)}`}>
                      Running the AI check
                    </span>
                  </div>
                  <div className="scan-step">
                    <div className={`step-dot ${stepClass(2)}`} />
                    <span className={`step-text ${stepClass(2)}`}>
                      Measuring what contributed to the score
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {result && <ResultsPanel result={result} inputText={inputSnapshot} />}
        </main>

        <footer className="footer">
          <p>
            Powered by a fine-tuned BERT model · Explainable AI · Final Year
            Project at Asia Pacific University of Technology and Innovation
          </p>
        </footer>
      </div>
    </>
  );
}