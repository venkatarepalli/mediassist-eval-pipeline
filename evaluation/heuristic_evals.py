"""
Heuristic Evaluations for MediAssist AI Evaluation Pipeline
------------------------------------------------------------
4 deterministic, rule-based checks that require no LLM call.
Runs as part of the same evaluation pipeline as RAGAS eval.

Checks:
  1. Citation check    : Every response includes at least one source citation
  2. RBAC refusal check: Restricted-role queries are refused, not answered
  3. Latency check     : Response latency stays under 10 seconds
  4. Empty answer check: No response contains empty or null answer field

Run: python -m evaluation.heuristic_evals
"""

import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/heuristic_evals.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
os.makedirs("reports", exist_ok=True)
RAGAS_RAW_PATH = "reports/ragas_raw_results.json"
HEURISTIC_RESULTS_PATH = "reports/heuristic_results.json"

# ── Thresholds ─────────────────────────────────────────────────────────────────
LATENCY_THRESHOLD_MS = 10000   # 10 seconds
MIN_ANSWER_LENGTH = 10         # minimum characters for a valid answer

# ── RBAC restricted combinations ───────────────────────────────────────────────
# These role+collection combos should always be refused
RBAC_RESTRICTED = [
    {"role": "nurse",      "collection": "billing"},
    {"role": "technician", "collection": "clinical"},
]

# ── Refusal phrases MediBot uses ───────────────────────────────────────────────
REFUSAL_PHRASES = [
    "i don't have",
    "i do not have",
    "i don't have access",
    "not have access",
    "insufficient",
    "unable to",
    "cannot provide",
    "no relevant information",
    "not able to answer",
    "outside my access",
]


def check_citation(result: dict) -> dict:
    """
    Check 1: Every successful response must include at least one source citation.
    A citation is defined as having at least one entry in the sources list.
    """
    if result["status"] != "success":
        return {
            "check": "citation_check",
            "passed": None,
            "reason": "skipped — MediBot returned no answer",
        }

    sources = result.get("sources", [])
    has_citation = len(sources) > 0

    return {
        "check": "citation_check",
        "passed": has_citation,
        "reason": (
            f"Found {len(sources)} source(s)"
            if has_citation
            else "No sources cited in response"
        ),
    }


def check_rbac_refusal(result: dict) -> dict:
    """
    Check 2: Restricted-role queries must be refused, not answered.
    A nurse asking billing questions must get a refusal.
    """
    role = result.get("role", "")
    collection = result.get("collection", "")

    # Check if this is a restricted combination
    is_restricted = any(
        r["role"] == role and r["collection"] == collection
        for r in RBAC_RESTRICTED
    )

    if not is_restricted:
        return {
            "check": "rbac_refusal_check",
            "passed": None,
            "reason": f"Not a restricted role/collection combo ({role}/{collection}) — skipped",
        }

    # Check if answer contains a refusal phrase
    answer_lower = result.get("answer", "").lower()
    was_refused = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)

    return {
        "check": "rbac_refusal_check",
        "passed": was_refused,
        "reason": (
            f"Correctly refused restricted query ({role} accessing {collection})"
            if was_refused
            else f"FAILED — {role} was given access to {collection} content"
        ),
    }


def check_latency(result: dict) -> dict:
    """
    Check 3: Response latency must be under 10 seconds (10,000ms).
    """
    if result["status"] != "success":
        return {
            "check": "latency_check",
            "passed": None,
            "reason": "skipped — MediBot returned no answer",
        }

    latency_ms = result.get("latency_ms", 0)
    passed = latency_ms <= LATENCY_THRESHOLD_MS

    return {
        "check": "latency_check",
        "passed": passed,
        "reason": (
            f"Latency {latency_ms:.0f}ms — within {LATENCY_THRESHOLD_MS}ms threshold"
            if passed
            else f"Latency {latency_ms:.0f}ms — EXCEEDED {LATENCY_THRESHOLD_MS}ms threshold"
        ),
    }


def check_empty_answer(result: dict) -> dict:
    """
    Check 4: No response contains an empty or null answer field.
    """
    answer = result.get("answer", None)

    if answer is None:
        return {
            "check": "empty_answer_check",
            "passed": False,
            "reason": "Answer field is null",
        }

    if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
        return {
            "check": "empty_answer_check",
            "passed": False,
            "reason": f"Answer too short ({len(answer.strip())} chars) — likely empty",
        }

    return {
        "check": "empty_answer_check",
        "passed": True,
        "reason": f"Answer has {len(answer.strip())} characters — valid",
    }


def run_heuristic_evals() -> dict:
    """
    Runs all 4 heuristic checks on MediBot's responses.
    Loads raw results from ragas_raw_results.json.
    """
    print("=" * 60)
    print("HEURISTIC EVALUATIONS")
    print("=" * 60)

    if not os.path.exists(RAGAS_RAW_PATH):
        print(f"❌ Raw results not found at {RAGAS_RAW_PATH}")
        print("   Run evaluation/ragas_eval.py first")
        return {}

    with open(RAGAS_RAW_PATH, encoding="utf-8") as f:
        raw_results = json.load(f)

    print(f"\n📋 Running heuristic checks on {len(raw_results)} responses...")
    print("-" * 60)

    per_question = []

    for r in raw_results:
        qid = r["id"]
        print(f"\n[{qid}] [{r['category']}] {r['question'][:65]}...")

        checks = [
            check_citation(r),
            check_rbac_refusal(r),
            check_latency(r),
            check_empty_answer(r),
        ]

        for check in checks:
            if check["passed"] is True:
                icon = "✅"
            elif check["passed"] is False:
                icon = "❌"
            else:
                icon = "⏭️ "
            print(f"  {icon} {check['check']}: {check['reason'][:70]}")

        per_question.append({
            "id": qid,
            "question": r["question"],
            "category": r["category"],
            "role": r["role"],
            "checks": checks,
        })

        logger.info(json.dumps({
            "event": "HEURISTIC_RESULT",
            "id": qid,
            "category": r["category"],
            "role": r["role"],
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }))

    # Aggregate results
    print("\n\n📊 Heuristic Evaluation Summary:")
    print("-" * 40)

    check_names = [
        "citation_check",
        "rbac_refusal_check",
        "latency_check",
        "empty_answer_check",
    ]

    aggregate = {}
    for check_name in check_names:
        all_checks = [
            check
            for r in per_question
            for check in r["checks"]
            if check["check"] == check_name and check["passed"] is not None
        ]
        passed = sum(1 for c in all_checks if c["passed"])
        total = len(all_checks)
        rate = round(passed / total, 4) if total > 0 else None

        aggregate[check_name] = {
            "passed": passed,
            "total": total,
            "pass_rate": rate,
        }

        status = "✅" if rate and rate >= 0.8 else "⚠️ " if rate and rate >= 0.5 else "❌"
        print(f"  {status} {check_name}: {passed}/{total} passed ({rate:.0%} pass rate)" if rate is not None else f"  ⏭️  {check_name}: no applicable results")

    output = {
        "aggregate": aggregate,
        "per_question": per_question,
        "thresholds": {
            "latency_ms": LATENCY_THRESHOLD_MS,
            "min_answer_length": MIN_ANSWER_LENGTH,
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(HEURISTIC_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Heuristic results saved to {HEURISTIC_RESULTS_PATH}")

    return output


if __name__ == "__main__":
    run_heuristic_evals()