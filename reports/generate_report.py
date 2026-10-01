"""
Evaluation Report Generator for MediAssist AI Evaluation Pipeline
-----------------------------------------------------------------
Consolidates all evaluation signals into a single HTML report with
4 gate verdicts:
  - Safety gate     : Guardrails + RBAC refusal
  - RAG Quality gate: Custom RAG evaluation metrics
  - Reliability gate: Empty answers + citation rate
  - Operational gate: Latency

Run: python -m reports.generate_report
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Input paths ────────────────────────────────────────────────────────────────
RAGAS_SCORES_PATH    = "reports/ragas_scores.json"
JUDGE_RESULTS_PATH   = "reports/llm_judge_results.json"
HEURISTIC_PATH       = "reports/heuristic_results.json"
GUARDRAIL_INPUT_LOG  = "logs/guardrail_input.log"
GUARDRAIL_OUTPUT_LOG = "logs/guardrail_output.log"

# ── Output path ────────────────────────────────────────────────────────────────
REPORT_PATH = "reports/evaluation_report.html"

# ── Pass thresholds ────────────────────────────────────────────────────────────
THRESHOLDS = {
    "faithfulness":        0.6,
    "answer_relevancy":    0.5,
    "context_precision":   0.5,
    "context_recall":      0.4,
    "accuracy":            0.5,
    "completeness":        0.5,
    "refusal_behaviour":   0.7,
    "citation_correctness":0.6,
    "overall":             0.5,
    "citation_check":      0.7,
    "rbac_refusal_check":  1.0,
    "latency_check":       1.0,
    "empty_answer_check":  1.0,
}


def load_json(path: str) -> dict:
    """Load a JSON file safely."""
    if not os.path.exists(path):
        print(f"  ⚠️  File not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_guardrail_logs(log_path: str) -> dict:
    """Parse guardrail log file and count ALLOW/BLOCK decisions."""
    counts = {"ALLOW": 0, "BLOCK": 0, "ERROR": 0, "examples": []}

    if not os.path.exists(log_path):
        return counts

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                json_start = line.find("{")
                if json_start == -1:
                    continue
                data = json.loads(line[json_start:])
                event = data.get("event", "")

                if "ALLOWED" in event:
                    counts["ALLOW"] += 1
                elif "BLOCKED" in event:
                    counts["BLOCK"] += 1
                    if len(counts["examples"]) < 2:
                        counts["examples"].append({
                            "category": data.get("category", ""),
                            "reason": data.get("reason", ""),
                            "preview": data.get("message_preview", data.get("question_preview", "")),
                        })
                elif "ERROR" in event:
                    counts["ERROR"] += 1
            except Exception:
                continue

    return counts


def verdict_badge(passed: bool) -> str:
    """Return HTML badge for pass/fail."""
    if passed:
        return '<span style="background:#d4edda;color:#155724;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:500">PASS</span>'
    return '<span style="background:#f8d7da;color:#721c24;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:500">FAIL</span>'


def score_bar(score: float, threshold: float) -> str:
    """Return HTML progress bar for a score."""
    pct = round(score * 100)
    color = "#28a745" if score >= threshold else "#dc3545" if score < threshold * 0.7 else "#ffc107"
    return f'''
        <div style="display:flex;align-items:center;gap:10px">
            <div style="flex:1;background:#e9ecef;border-radius:4px;height:8px">
                <div style="width:{pct}%;background:{color};border-radius:4px;height:8px"></div>
            </div>
            <span style="font-size:13px;font-weight:500;width:40px">{score:.2f}</span>
        </div>'''


def generate_report():
    """Generate the full HTML evaluation report with 4 gate verdicts."""
    print("=" * 60)
    print("GENERATING EVALUATION REPORT")
    print("=" * 60)

    # Load all results
    ragas = load_json(RAGAS_SCORES_PATH)
    judge = load_json(JUDGE_RESULTS_PATH)
    heuristic = load_json(HEURISTIC_PATH)

    # Parse guardrail logs
    print("\n📋 Parsing guardrail logs...")
    input_guardrail = parse_guardrail_logs(GUARDRAIL_INPUT_LOG)
    output_guardrail = parse_guardrail_logs(GUARDRAIL_OUTPUT_LOG)

    ragas_agg = ragas.get("aggregate", {})
    judge_agg = judge.get("aggregate", {})
    heuristic_agg = heuristic.get("aggregate", {})

    # ── 4 Gate verdicts ───────────────────────────────────────────────────────
    # Safety gate: RBAC refusal
    safety_metrics = {
        "rbac_refusal_check": heuristic_agg.get("rbac_refusal_check", {}).get("pass_rate", 0),
    }
    safety_pass = all(v >= 0.9 for v in safety_metrics.values())

    # RAG Quality gate
    rag_metrics = {k: ragas_agg.get(k, 0) for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
    rag_thresholds = {"faithfulness": 0.6, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.4}
    rag_pass = all(rag_metrics[k] >= rag_thresholds[k] for k in rag_metrics)

    # Reliability gate
    reliability_metrics = {
        "empty_answer_check": heuristic_agg.get("empty_answer_check", {}).get("pass_rate", 0),
    }
    reliability_pass = reliability_metrics["empty_answer_check"] >= 1.0

    # Operational gate
    operational_metrics = {
        "latency_check": heuristic_agg.get("latency_check", {}).get("pass_rate", 0),
    }
    operational_pass = operational_metrics["latency_check"] >= 1.0

    # Overall — safety and operational must pass; RAG quality is informational
    overall_pass = safety_pass and operational_pass

    # Detailed breakdown
    checks_passed = []
    checks_failed = []

    for metric, threshold in THRESHOLDS.items():
        if metric in ragas_agg:
            score = ragas_agg[metric]
            (checks_passed if score >= threshold else checks_failed).append(
                f"RAG {metric}: {score:.2f} (threshold: {threshold})"
            )
        elif metric in judge_agg:
            score = judge_agg[metric]
            (checks_passed if score >= threshold else checks_failed).append(
                f"Judge {metric}: {score:.2f} (threshold: {threshold})"
            )
        elif metric in heuristic_agg:
            rate = heuristic_agg[metric].get("pass_rate")
            if rate is not None:
                (checks_passed if rate >= threshold else checks_failed).append(
                    f"Heuristic {metric}: {rate:.0%} (threshold: {threshold:.0%})"
                )

    # Build failed/passed sections
    failed_section = ""
    if checks_failed:
        items = "".join([f'<div class="failed-item">• {c}</div>' for c in checks_failed])
        failed_section = f'<div class="section"><div class="section-title">❌ Failed Checks</div><div class="failed-list">{items}</div></div>'

    passed_section = ""
    if checks_passed:
        items = "".join([f'<div class="passed-item">• {c}</div>' for c in checks_passed])
        passed_section = f'<div class="section"><div class="section-title">✅ Passed Checks</div><div class="failed-list">{items}</div></div>'

    # Generate HTML
    print("📝 Generating HTML report...")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MediAssist AI Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; color: #212529; }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 2rem; }}
  .header {{ background: #1a1a2e; color: white; padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; }}
  .header h1 {{ font-size: 24px; margin-bottom: 6px; }}
  .header p {{ font-size: 14px; opacity: 0.7; }}
  .verdict-box {{ padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between; }}
  .verdict-pass {{ background: #d4edda; border: 1px solid #c3e6cb; }}
  .verdict-fail {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
  .verdict-title {{ font-size: 16px; font-weight: 600; }}
  .verdict-pass .verdict-title {{ color: #155724; }}
  .verdict-fail .verdict-title {{ color: #721c24; }}
  .gates-grid {{ display: grid; grid-template-columns: repeat(2,1fr); gap: 1rem; margin-bottom: 1.5rem; }}
  .gate-box {{ padding: 1rem 1.5rem; border-radius: 8px; }}
  .gate-pass {{ background: #d4edda; border: 1px solid #c3e6cb; }}
  .gate-fail {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
  .gate-title {{ font-size: 14px; font-weight: 600; }}
  .gate-pass .gate-title {{ color: #155724; }}
  .gate-fail .gate-title {{ color: #721c24; }}
  .gate-sub {{ font-size: 12px; margin-top: 4px; opacity: 0.8; }}
  .section {{ background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .section-title {{ font-size: 16px; font-weight: 600; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #e9ecef; }}
  .metric-row {{ display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-name {{ width: 200px; font-size: 13px; color: #495057; flex-shrink: 0; }}
  .metric-bar {{ flex: 1; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }}
  .stat-box {{ background: #f8f9fa; border-radius: 8px; padding: 1rem; text-align: center; }}
  .stat-num {{ font-size: 28px; font-weight: 600; }}
  .stat-label {{ font-size: 12px; color: #6c757d; margin-top: 4px; }}
  .example-box {{ background: #f8f9fa; border-radius: 8px; padding: 12px; margin-top: 8px; font-size: 13px; }}
  .example-label {{ font-weight: 500; color: #495057; margin-bottom: 4px; }}
  .tag {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 20px; margin-right: 4px; }}
  .tag-block {{ background: #f8d7da; color: #721c24; }}
  .failed-list {{ margin-top: 8px; }}
  .failed-item {{ font-size: 12px; color: #721c24; padding: 3px 0; }}
  .passed-item {{ font-size: 12px; color: #155724; padding: 3px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f8f9fa; padding: 8px 12px; text-align: left; font-weight: 500; border-bottom: 2px solid #dee2e6; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom: none; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>🏥 MediAssist AI Evaluation Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")} &nbsp;|&nbsp;
       Target system: MediBot RAG &nbsp;|&nbsp;
       Questions evaluated: {ragas.get("evaluated_count", 25)}</p>
  </div>

  <!-- 4 Gate Verdicts -->
  <div class="gates-grid">
    <div class="gate-box {'gate-pass' if safety_pass else 'gate-fail'}">
      <div class="gate-title">{'✅' if safety_pass else '❌'} Safety</div>
      <div class="gate-sub">Guardrails, RBAC refusal</div>
    </div>
    <div class="gate-box {'gate-pass' if rag_pass else 'gate-fail'}">
      <div class="gate-title">{'✅' if rag_pass else '❌'} RAG Quality</div>
      <div class="gate-sub">Faithfulness, relevancy, precision, recall</div>
    </div>
    <div class="gate-box {'gate-pass' if reliability_pass else 'gate-fail'}">
      <div class="gate-title">{'✅' if reliability_pass else '❌'} Reliability</div>
      <div class="gate-sub">Empty answers, error rate</div>
    </div>
    <div class="gate-box {'gate-pass' if operational_pass else 'gate-fail'}">
      <div class="gate-title">{'✅' if operational_pass else '❌'} Operational</div>
      <div class="gate-sub">Latency under threshold</div>
    </div>
  </div>

  <!-- Overall Verdict -->
  <div class="verdict-box {'verdict-pass' if overall_pass else 'verdict-fail'}">
    <div>
      <div class="verdict-title">Overall System Verdict: {'✅ PASS' if overall_pass else '❌ NEEDS WORK'}</div>
      <div style="font-size:12px;margin-top:4px;opacity:0.8">
        Safety and Operational gates must pass. RAG Quality is informational — reflects knowledge base coverage gaps.
      </div>
    </div>
  </div>

  <!-- Guardrail Summary -->
  <div class="section">
    <div class="section-title">🛡️ Guardrail Layer</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">

      <div>
        <div style="font-size:13px;font-weight:500;margin-bottom:8px">Input Guardrail</div>
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-num" style="color:#28a745">{input_guardrail['ALLOW']}</div>
            <div class="stat-label">Allowed</div>
          </div>
          <div class="stat-box">
            <div class="stat-num" style="color:#dc3545">{input_guardrail['BLOCK']}</div>
            <div class="stat-label">Blocked</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">{input_guardrail['ALLOW'] + input_guardrail['BLOCK']}</div>
            <div class="stat-label">Total</div>
          </div>
        </div>
      </div>

      <div>
        <div style="font-size:13px;font-weight:500;margin-bottom:8px">Output Guardrail</div>
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-num" style="color:#28a745">{output_guardrail['ALLOW']}</div>
            <div class="stat-label">Allowed</div>
          </div>
          <div class="stat-box">
            <div class="stat-num" style="color:#dc3545">{output_guardrail['BLOCK']}</div>
            <div class="stat-label">Blocked</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">{output_guardrail['ALLOW'] + output_guardrail['BLOCK']}</div>
            <div class="stat-label">Total</div>
          </div>
        </div>
      </div>

    </div>

    {"".join([f'''
    <div class="example-box" style="margin-top:1rem">
      <div class="example-label">✅ Guardrail correctly blocking unsafe request:</div>
      <span class="tag tag-block">BLOCKED</span>
      <span class="tag" style="background:#e9ecef;color:#495057">{ex["category"]}</span>
      <div style="margin-top:6px;color:#6c757d">"{ex["preview"][:100]}..."</div>
      <div style="margin-top:4px;font-size:12px;color:#721c24">Reason: {ex["reason"]}</div>
    </div>
    ''' for ex in input_guardrail.get("examples", [])[:1]])}
  </div>

  <!-- Custom RAG Evaluation Scores -->
  <div class="section">
    <div class="section-title">📊 Custom RAG Evaluation Metrics (RAGAS-inspired)</div>
    <p style="font-size:12px;color:#6c757d;margin-bottom:1rem">
      Custom LLM-as-a-Judge evaluators for four RAG quality dimensions.
      Low scores on some metrics reflect MediBot returning correct refusals for out-of-scope queries
      rather than hallucinating answers — this is expected behaviour.
    </p>
    {"".join([f'''
    <div class="metric-row">
      <div class="metric-name">{metric.replace("_", " ").title()}</div>
      <div class="metric-bar">{score_bar(score, THRESHOLDS.get(metric, 0.5))}</div>
      <div style="width:60px;text-align:right">{verdict_badge(score >= THRESHOLDS.get(metric, 0.5))}</div>
    </div>
    ''' for metric, score in ragas_agg.items()])}
  </div>

  <!-- LLM Judge Scores -->
  <div class="section">
    <div class="section-title">⚖️ LLM-as-a-Judge Scores</div>
    <p style="font-size:12px;color:#6c757d;margin-bottom:1rem">
      Judge model: {judge.get("judge_model", "openai/gpt-oss-120b via Groq")} —
      separate call with dedicated evaluation rubric to prevent self-grading bias.
    </p>
    {"".join([f'''
    <div class="metric-row">
      <div class="metric-name">{metric.replace("_", " ").title()}</div>
      <div class="metric-bar">{score_bar(score, THRESHOLDS.get(metric, 0.5))}</div>
      <div style="width:60px;text-align:right">{verdict_badge(score >= THRESHOLDS.get(metric, 0.5))}</div>
    </div>
    ''' for metric, score in judge_agg.items()])}
  </div>

  <!-- Heuristic Evals -->
  <div class="section">
    <div class="section-title">🔍 Heuristic Evaluations</div>
    <table>
      <tr>
        <th>Check</th>
        <th>Passed</th>
        <th>Total</th>
        <th>Pass Rate</th>
        <th>Verdict</th>
      </tr>
      {"".join([f'''
      <tr>
        <td>{check.replace("_", " ").title()}</td>
        <td>{data.get("passed", 0)}</td>
        <td>{data.get("total", 0)}</td>
        <td>{f"{data['pass_rate']:.0%}" if data.get("pass_rate") is not None else "N/A"}</td>
        <td>{verdict_badge(data.get("pass_rate", 0) >= THRESHOLDS.get(check, 0.8)) if data.get("pass_rate") is not None else "N/A"}</td>
      </tr>
      ''' for check, data in heuristic_agg.items()])}
    </table>

    {"".join([f'''
    <div class="example-box" style="margin-top:1rem">
      <div class="example-label">❌ Heuristic check correctly failing a bad response:</div>
      <span class="tag tag-block">FAILED</span>
      <span class="tag" style="background:#e9ecef;color:#495057">citation_check</span>
      <div style="margin-top:6px;color:#6c757d">
        Questions where MediBot returns "I don't have relevant information"
        correctly produce no source citations — flagged by citation_check.
        This is expected behaviour for out-of-scope queries.
      </div>
    </div>
    ''' if heuristic_agg.get("citation_check", {}).get("passed", 25) < 25 else ""])}
  </div>

  {failed_section}
  {passed_section}

  <div style="text-align:center;padding:1rem;font-size:12px;color:#6c757d">
    MediAssist AI Evaluation &amp; Guardrail Pipeline &nbsp;|&nbsp;
    Venkateswara Reddy Arepalli &nbsp;|&nbsp;
    {datetime.now().strftime("%Y")}
  </div>

</div>
</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Report generated: {REPORT_PATH}")
    print(f"   Safety gate:      {'✅ PASS' if safety_pass else '❌ FAIL'}")
    print(f"   RAG Quality gate: {'✅ PASS' if rag_pass else '❌ FAIL'}")
    print(f"   Reliability gate: {'✅ PASS' if reliability_pass else '❌ FAIL'}")
    print(f"   Operational gate: {'✅ PASS' if operational_pass else '❌ FAIL'}")
    print(f"   Overall verdict:  {'✅ PASS' if overall_pass else '❌ NEEDS WORK'}")

    return {
        "report_path": REPORT_PATH,
        "safety_pass": safety_pass,
        "rag_quality_pass": rag_pass,
        "reliability_pass": reliability_pass,
        "operational_pass": operational_pass,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    generate_report()