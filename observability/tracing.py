"""
Observability & Tracing for MediAssist AI Evaluation Pipeline
-------------------------------------------------------------
Instruments the full request path with LangSmith tracing.
Every request produces a trace viewable at smith.langchain.com.
Captures: latency, token usage, guardrail decisions, retrieval path.
"""

import json
import logging
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_helpers import traceable

load_dotenv()

# ── LangSmith setup ────────────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "mediassist-eval-pipeline")

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/observability.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("observability")

# ── LangSmith client ───────────────────────────────────────────────────────────
langsmith_client = Client()


def log_request_metric(
    question: str,
    role: str,
    input_verdict: str,
    output_verdict: str,
    latency_ms: float,
    retrieval_type: str = "unknown",
    tokens_used: int = 0,
):
    """
    Logs a structured metric line for every request.
    Queryable from logs/observability.log.
    """
    event = {
        "event": "REQUEST_METRIC",
        "timestamp": datetime.utcnow().isoformat(),
        "role": role,
        "question_preview": question[:100],
        "input_guardrail": input_verdict,
        "output_guardrail": output_verdict,
        "retrieval_type": retrieval_type,
        "latency_ms": round(latency_ms, 2),
        "tokens_used": tokens_used,
    }
    logger.info(json.dumps(event))


def log_guardrail_decision(
    stage: str,
    verdict: str,
    category: str,
    reason: str,
    role: str,
):
    """
    Logs a guardrail decision as a structured event.
    stage: INPUT or OUTPUT
    """
    event = {
        "event": f"GUARDRAIL_{stage}_{verdict}",
        "timestamp": datetime.utcnow().isoformat(),
        "stage": stage,
        "verdict": verdict,
        "category": category,
        "reason": reason,
        "role": role,
    }
    if verdict == "ALLOW":
        logger.info(json.dumps(event))
    else:
        logger.warning(json.dumps(event))


@traceable(name="mediassist-full-request")
def trace_full_request(
    question: str,
    role: str,
    input_result: dict,
    medibot_response: dict | None,
    output_result: dict | None,
    latency_ms: float,
):
    """
    Creates a LangSmith trace for the full request path.
    Opens as a single inspectable trace at smith.langchain.com.

    Args:
        question: original user question
        role: JWT role of the user
        input_result: verdict from input guardrail
        medibot_response: raw response from MediBot (None if input blocked)
        output_result: verdict from output guardrail (None if input blocked)
        latency_ms: total end-to-end latency in milliseconds
    """
    trace_data = {
        "question": question,
        "role": role,
        "input_guardrail": {
            "verdict": input_result.get("verdict"),
            "category": input_result.get("category"),
            "reason": input_result.get("reason"),
        },
        "medibot_response": medibot_response,
        "output_guardrail": {
            "verdict": output_result.get("verdict") if output_result else "NOT_RUN",
            "category": output_result.get("category") if output_result else "NOT_RUN",
            "reason": output_result.get("reason") if output_result else "NOT_RUN",
        } if output_result else {"verdict": "NOT_RUN"},
        "latency_ms": round(latency_ms, 2),
        "final_outcome": (
            "BLOCKED_AT_INPUT" if not input_result.get("allowed")
            else "BLOCKED_AT_OUTPUT" if output_result and not output_result.get("allowed")
            else "ALLOWED"
        ),
    }

    # Log metrics
    log_request_metric(
        question=question,
        role=role,
        input_verdict=input_result.get("verdict", "UNKNOWN"),
        output_verdict=output_result.get("verdict", "NOT_RUN") if output_result else "NOT_RUN",
        latency_ms=latency_ms,
        retrieval_type=medibot_response.get("retrieval_type", "unknown") if medibot_response else "blocked",
        tokens_used=0,
    )

    return trace_data