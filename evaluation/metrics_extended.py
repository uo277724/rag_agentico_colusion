# evaluation/metrics_extended.py
"""
Módulo de métricas extendidas inspirado en el artículo de evaluación de LLMs.
Calcula métricas cuantitativas adicionales para analizar la calidad y eficiencia del sistema RAG.

Incluye:
- Execution time
- Cost (estimado)
- Narrative length
- Informativeness (número de puntos de información distintos)
- Accuracy (si se dispone de etiquetas)
- Accurate Narrative Information Density (ANID)
- Métricas de legibilidad: FRE, FKGL, ARI
"""

import time
import re
import textstat
import numpy as np


# ============================================================
# FUNCIONES PRINCIPALES
# ============================================================

def compute_extended_metrics(answer: str, judge_result: dict, start_time=None, end_time=None, model_name="gpt-4o"):
    """
    Calcula métricas extendidas del sistema RAG, inspiradas en el artículo.
    
    Parámetros:
        answer (str): Respuesta generada por el modelo.
        judge_result (dict): Resultado del módulo judge_paper (clarity, relevance, etc.).
        start_time (float, opcional): Timestamp al inicio de la generación.
        end_time (float, opcional): Timestamp al final de la generación.
        model_name (str): Nombre del modelo (para estimar el coste).

    Retorna:
        dict con las métricas calculadas.
    """

    # -------------------------
    # Execution time
    # -------------------------
    if start_time and end_time:
        execution_time = round(end_time - start_time, 3)
    else:
        execution_time = None

    # -------------------------
    # Cost estimation (USD) — actualizado para GPT-4o y GPT-4o-mini (2025)
    # -------------------------
    words = len(answer.split())
    est_tokens = int(words / 0.75)

    # Precios oficiales (USD por token)
    prices = {
        "gpt-4o": {"input": 5.00e-6, "output": 15.00e-6},
        "gpt-4o-mini": {"input": 0.60e-6, "output": 2.40e-6},
    }

    m = next((k for k in prices if k in model_name.lower()), "gpt-4o-mini")
    cost_per_token = np.mean(list(prices[m].values()))

    cost_usd = round(est_tokens * cost_per_token, 5)
    cost_cents = round(cost_usd * 100, 3)

    # -------------------------
    # Narrative length
    # -------------------------
    narrative_length = words

    # -------------------------
    # Informativeness (conteo de ideas)
    # -------------------------
    # Aproximación: número de oraciones que contienen un verbo técnico
    technical_verbs = ["controla", "mide", "regula", "indica", "opera", "calienta",
                       "ajusta", "funciona", "representa", "activa", "desactiva", "supervisa"]
    sentences = re.split(r'[.!?]+', answer)
    info_points = sum(any(v in s.lower() for v in technical_verbs) for s in sentences if len(s.strip()) > 5)

    # -------------------------
    # Accuracy (si hubiera etiquetas de verdad)
    # -------------------------
    # Placeholder: si judge_result trae Insightfulness y Relevance, se usa como proxy
    factual_accuracy = round(
        np.mean([judge_result.get("Relevance", 0), judge_result.get("Insightfulness", 0)]) / 4, 3
    )

    # -------------------------
    # ANID (Accurate Narrative Information Density)
    # -------------------------
    # Correct distinct info points por 100 palabras
    anid = round((info_points / max(words, 1)) * 100, 3)

    # -------------------------
    # Readability metrics (adaptadas a español)
    # -------------------------
    try:
        fre = textstat.flesch_reading_ease(answer)
        fkgl = textstat.flesch_kincaid_grade(answer)
        ari = textstat.automated_readability_index(answer)

        # Ajuste heurístico para textos técnicos en español:
        # FRE puede ser negativo o bajo; normalizamos a [0, 100]
        fre_norm = max(0, min(100, fre))
        # FKGL y ARI son grados escolares → invertimos para que "más fácil = más alto"
        clarity_grade = max(0, min(100, 100 - ((fkgl + ari) / 2)))
    except Exception:
        fre_norm, fkgl, ari, clarity_grade = 0, 0, 0, 0

    # -------------------------
    # Resumen consolidado
    # -------------------------
    extended_metrics = {
        "execution_time_sec": execution_time,
        "cost_cents": cost_cents,
        "narrative_length_words": narrative_length,
        "informativeness_points": info_points,
        "accuracy_est": factual_accuracy,
        "anid": anid,
        "flesch_reading_ease": round(fre_norm, 2),
        "flesch_kincaid_grade": round(fkgl, 2),
        "automated_readability_index": round(ari, 2),
        "clarity_grade": round(clarity_grade, 2),
        "model_name": model_name,
    }

    # Métrica global combinada (ponderada)
    extended_metrics["global_quality_index"] = round((
        0.25 * factual_accuracy +
        0.25 * (anid / 100) +
        0.25 * (judge_result.get("Overall_score", 0) / 4) +
        0.25 * (clarity_grade / 100)
    ), 3)

    return extended_metrics
