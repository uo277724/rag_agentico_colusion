"""
Métricas extendidas para sistemas RAG analíticos y conservadores.
Diseñadas para entornos auditables (contratación pública, colusión, compliance).
"""

import time
import numpy as np
import textstat


# ============================================================
# MÉTRICAS PRINCIPALES
# ============================================================

def compute_extended_metrics(
    answer: str,
    judge_result: dict,
    start_time=None,
    end_time=None,
    model_name="gpt-4o-mini",
):
    # --------------------------------------------------
    # Execution time
    # --------------------------------------------------
    execution_time = (
        round(end_time - start_time, 3)
        if start_time and end_time
        else None
    )

    # --------------------------------------------------
    # Cost estimation (aprox.)
    # --------------------------------------------------
    words = len(answer.split())
    est_tokens = int(words / 0.75)

    prices = {
        "gpt-4o": 10.00e-6,
        "gpt-4o-mini": 1.50e-6,
    }

    m = next((k for k in prices if k in model_name.lower()), "gpt-4o-mini")
    cost_usd = est_tokens * prices[m]
    cost_cents = round(cost_usd * 100, 4)

    # --------------------------------------------------
    # Narrative length
    # --------------------------------------------------
    narrative_length = words

    # --------------------------------------------------
    # Faithfulness & Traceability (from judge)
    # --------------------------------------------------
    faithfulness = judge_result.get("Faithfulness", 0) / 4
    traceability = judge_result.get("Traceability", 0) / 4

    issue_types = set(judge_result.get("Issue_types", []))

    # Penalización explícita por inferencias
    inference_penalty = 0.0
    if "unsupported_inference" in issue_types:
        inference_penalty = 0.3
    if "analysis_error" in issue_types or "wrong_calculation" in issue_types:
        inference_penalty = 0.6

    safe_faithfulness = max(0.0, faithfulness - inference_penalty)

    # --------------------------------------------------
    # TSID — Traceable Statement Information Density
    # --------------------------------------------------
    # Proxy conservador:
    # densidad = faithfulness * traceability / longitud
    tsid = round(
        (safe_faithfulness * traceability) * 100 / max(words, 1),
        4
    )

    # --------------------------------------------------
    # Readability (secundaria)
    # --------------------------------------------------
    try:
        fre = max(0, min(100, textstat.flesch_reading_ease(answer)))
        fkgl = textstat.flesch_kincaid_grade(answer)
        ari = textstat.automated_readability_index(answer)
        clarity_grade = max(0, min(100, 100 - ((fkgl + ari) / 2)))
    except Exception:
        fre, fkgl, ari, clarity_grade = 0, 0, 0, 0

    # --------------------------------------------------
    # Global Quality Index (conservador)
    # --------------------------------------------------
    global_quality_index = round(
        0.4 * safe_faithfulness +
        0.3 * traceability +
        0.2 * (clarity_grade / 100) +
        0.1 * (1 / max(words, 1)),
        4
    )

    return {
        "execution_time_sec": execution_time,
        "cost_cents": cost_cents,
        "narrative_length_words": narrative_length,
        "faithfulness": round(faithfulness, 3),
        "traceability": round(traceability, 3),
        "safe_faithfulness": round(safe_faithfulness, 3),
        "tsid": tsid,
        "flesch_reading_ease": round(fre, 2),
        "clarity_grade": round(clarity_grade, 2),
        "issue_types": list(issue_types),
        "model_name": model_name,
        "global_quality_index": global_quality_index,
    }
