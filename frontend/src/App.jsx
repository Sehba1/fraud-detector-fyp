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
  if (label === "fraudulent" && tier === "high") return "LIKELY A SCAM";
  if (label === "fraudulent" && tier === "medium") return "PROCEED WITH CAUTION";
  return "LOOKS GENUINE";
}

function getVerdictSub(label, tier) {
  if (label === "fraudulent" && tier === "high")
    return "This posting shows strong warning signs. Be very careful.";
  if (label === "fraudulent" && tier === "medium")
    return "Some warning signs were detected. Take a closer look before you apply.";
  return "No major red flags were found in this posting.";
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
          This posting has the hallmarks of a job scam.
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
        This looks like a real job posting.{" "}
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

function strengthOf(weight, maxWeight) {
  const ratio = Math.abs(weight) / maxWeight;
  if (ratio > 0.66) return "high";
  if (ratio > 0.33) return "med";
  return "low";
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

function PatternCard({ category, tone }) {
  const icon = PATTERN_ICONS[category.id] || "ti-point";
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
      <p className="pattern-explanation">{category.explanation}</p>
    </div>
  );
}

function PatternAnalysisSection({ analysis, label, riskClass }) {
  const warnTone = riskClass === "medium" ? "caution" : "warn";
  const hasWarnings = analysis.warning.length > 0;
  const hasGenuine = analysis.genuine.length > 0;
  const warnBannerClass = riskClass === "low" ? "medium" : riskClass;

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
            {label === "genuine" && (
              <p className="layer-note">
                <i className="ti ti-info-circle" aria-hidden="true" />
                These patterns appear in the text, but the AI model still judged
                the overall posting as genuine. Review them yourself before
                applying.
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
                The AI model classified this posting as fraudulent from its
                overall language, even though none of the known scam patterns
                were matched. The AI decision tab shows what influenced the
                model.
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
    wordMap[word.toLowerCase()] = weight;
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
          const kind = isWarning ? "Warning word" : "Safe word";
          return (
            <mark
              key={i}
              className={hlClass}
              title={`${kind}, impact: ${strengthOf(weight, maxWeight)}`}
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
  const [view, setView] = useState("patterns");
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

  const warnWords = result.explanation
    .filter((e) => e.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 5);

  const safeWords = result.explanation
    .filter((e) => e.weight < 0)
    .sort((a, b) => a.weight - b.weight)
    .slice(0, 5);

  const maxWeight = Math.max(
    ...result.explanation.map((e) => Math.abs(e.weight)),
    0.0001
  );

  const barWidth = (weight) =>
    `${Math.max((Math.abs(weight) / maxWeight) * 100, 8)}%`;

  const warnFill = riskClass === "medium" ? "caution" : "warn";
  const warnTitleClass = riskClass === "medium" ? "caution" : "warn";
  const warnIcon = riskClass === "medium" ? "ti-alert-circle" : "ti-flame";
  const warnHeading = riskClass === "low" ? "Minor concerns" : "Warning words";
  const highlightTitle =
    riskClass === "low" ? "Words that stood out" : "Words that raised the alarm";

  const hasOcr = result.input_type === "image" && result.ocr_text;
  const displayText = hasOcr ? result.ocr_text : inputText;

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
                {(result.fraud_probability * 100).toFixed(1)}%
              </span>
              <span className="metric-lbl">Scam chance</span>
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

        <div className="result-tabs">
          <button
            className={`rtab-btn ${view === "patterns" ? "active" : ""}`}
            onClick={() => setView("patterns")}
          >
            <i className="ti ti-shield-search" aria-hidden="true" />
            Scam check
          </button>
          <button
            className={`rtab-btn ${view === "model" ? "active" : ""}`}
            onClick={() => setView("model")}
          >
            <i className="ti ti-brain" aria-hidden="true" />
            AI decision
          </button>
          {hasOcr && (
            <button
              className={`rtab-btn ${view === "ocr" ? "active" : ""}`}
              onClick={() => setView("ocr")}
            >
              <i className="ti ti-camera" aria-hidden="true" />
              Picture text
            </button>
          )}
        </div>

        {view === "patterns" && result.pattern_analysis && (
          <div className="tab-pane">
            <PatternAnalysisSection
              analysis={result.pattern_analysis}
              label={result.label}
              riskClass={riskClass}
            />
          </div>
        )}

        {view === "model" && (
          <div className="tab-pane">
            <div className="result-section">
              <div className="section-title">
                <i className="ti ti-highlight" aria-hidden="true" />
                {highlightTitle}
              </div>
              <div className="hl-legend">
                <div className="hl-leg">
                  <div
                    className={`hl-dot ${riskClass === "medium" ? "caution" : "warn"}`}
                  />
                  Warning word, made it look like a scam
                </div>
                <div className="hl-leg">
                  <div className="hl-dot safe" />
                  Safe word, made it look genuine
                </div>
              </div>
              <HighlightedText
                text={displayText}
                explanation={result.explanation}
                riskClass={riskClass}
              />
            </div>

            <div className="result-section">
              <div className="section-title">
                <i className="ti ti-chart-bar" aria-hidden="true" />
                Word-by-word breakdown
              </div>
              <div className="signals-grid">
                <div>
                  <div className={`signal-col-title ${warnTitleClass}`}>
                    <i className={`ti ${warnIcon}`} aria-hidden="true" />
                    {warnHeading}
                  </div>
                  {warnWords.length > 0 ? (
                    warnWords.map(({ word, weight }) => (
                      <div key={word} className="signal-row">
                        <span className="signal-word">{word}</span>
                        <div className="signal-track">
                          <div
                            className={`signal-fill ${warnFill}`}
                            style={{ width: barWidth(weight) }}
                          />
                        </div>
                        <span className="signal-strength">
                          {strengthOf(weight, maxWeight)}
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="signal-empty">None found</p>
                  )}
                </div>
                <div>
                  <div className="signal-col-title safe">
                    <i className="ti ti-shield-check" aria-hidden="true" />
                    Safe words
                  </div>
                  {safeWords.length > 0 ? (
                    safeWords.map(({ word, weight }) => (
                      <div key={word} className="signal-row">
                        <span className="signal-word">{word}</span>
                        <div className="signal-track">
                          <div
                            className="signal-fill safe"
                            style={{ width: barWidth(weight) }}
                          />
                        </div>
                        <span className="signal-strength">
                          {strengthOf(weight, maxWeight)}
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="signal-empty">None found</p>
                  )}
                </div>
              </div>
              <div className="signal-note">
                <i className="ti ti-info-circle" aria-hidden="true" />
                This shows how the AI model itself weighed each word when making
                its decision. Longer bars mean a bigger impact. This is separate
                from the Scam check tab.
              </div>
            </div>
          </div>
        )}

        {view === "ocr" && hasOcr && (
          <div className="tab-pane">
            <div className="result-section">
              <div className="section-title">
                <i className="ti ti-camera" aria-hidden="true" />
                Text we read from your picture
              </div>
              <div className="ocr-block">
                <p className="ocr-text">{result.ocr_text}</p>
              </div>
            </div>
          </div>
        )}
      </div>
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
                Takes about 1 to 3 minutes
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
                      Building the explanation for each word
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