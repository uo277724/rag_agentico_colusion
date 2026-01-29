# evaluation/metrics.py

import numpy as np
import textstat
import json
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


# ============================================================
# MÉTRICAS CUANTITATIVAS BASADAS EN EMBEDDINGS Y LEGIBILIDAD
# ============================================================

def compute_numeric_metrics(question, context, answer, embedder):
    """
    Calcula métricas cuantitativas deterministas basadas en embeddings y legibilidad.
    """
    chunks = [c.strip() for c in context.split("\n---\n") if c.strip()]
    if not chunks:
        return {"factual_accuracy": 0.0, "informativeness": 0.0, "clarity": 0.0, "method": "numeric"}

    context_vecs = embedder.embed_texts(chunks)
    answer_vec = embedder.embed_texts([answer])[0]

    # --- Factual Accuracy ---
    mean_context = np.mean(context_vecs, axis=0)
    factual_accuracy = float(cosine_similarity([mean_context], [answer_vec])[0][0])

    # --- Informativeness ---
    sims = cosine_similarity(context_vecs, [answer_vec]).flatten()
    informativeness = float(np.mean(sims))

    # --- Clarity ---
    try:
        readability = textstat.flesch_reading_ease(answer)
        clarity = min(1.0, max(0.0, readability / 100))
    except Exception:
        clarity = 0.5

    return {
        "factual_accuracy": round(factual_accuracy, 3),
        "informativeness": round(informativeness, 3),
        "clarity": round(clarity, 3),
        "method": "numeric"
    }


# ============================================================
# MÉTRICAS CUALITATIVAS BASADAS EN LLM (JUEZ SEMÁNTICO)
# ============================================================

def compute_llm_metrics(question, context, answer, model_name="gpt-4o-mini"):
    """
    Usa un modelo LLM para juzgar la calidad semántica de la respuesta.
    Devuelve valores entre 0 y 1 para factual_accuracy, informativeness y clarity.
    """
    from openai import OpenAI
    import re

    client = OpenAI()

    # Seguridad: proteger contra entradas vacías o nulas
    question = question or "Pregunta no disponible"
    answer = answer or "Sin respuesta generada"
    context = (context or "Sin contexto disponible")[:6000]  # truncar contexto largo

    prompt = f"""
Evalúa la calidad de la siguiente respuesta técnica basándote únicamente en el contexto y la pregunta.
Responde EXCLUSIVAMENTE con un objeto JSON válido con tres campos numéricos entre 0 y 1:
{{
  "factual_accuracy": <float>,
  "informativeness": <float>,
  "clarity": <float>
}}

[PREGUNTA]
{question}

[CONTEXTO]
{context}

[RESPUESTA]
{answer}
    """

    try:
        # Intento 1: forzar salida en formato JSON nativo
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Eres un juez técnico experto en documentación industrial."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"}  # <--- fuerza formato JSON
        )
        text = response.choices[0].message.content.strip()
        scores = json.loads(text)

    except Exception:
        # Intento 2: recuperación de JSON dentro de texto libre
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Eres un juez técnico experto en documentación industrial."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=400
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                scores = json.loads(match.group(0))
            else:
                raise ValueError("No JSON found in response.")
        except Exception as e:
            print(f"[LLM-JUDGE ERROR] {e}")
            print(f"[LLM-JUDGE RAW RESPONSE] {text if 'text' in locals() else '(no response)'}")
            scores = {"factual_accuracy": 0.0, "informativeness": 0.0, "clarity": 0.0}

    scores["method"] = "llm"
    return scores


# ============================================================
# AGREGADOR DE RESULTADOS
# ============================================================

def aggregate_metrics(numeric_scores, llm_scores, weights=None):
    """
    Fusiona los resultados numéricos y LLM en un score híbrido.
    """
    if weights is None:
        weights = {"numeric": 0.6, "llm": 0.4}

    hybrid = {}
    for key in ["factual_accuracy", "informativeness", "clarity"]:
        hybrid[key] = round(
            weights["numeric"] * numeric_scores[key] + weights["llm"] * llm_scores[key],
            3
        )

    hybrid["method"] = "hybrid"
    hybrid["weights"] = weights
    hybrid["global_score"] = round(
        0.5 * hybrid["factual_accuracy"] +
        0.3 * hybrid["informativeness"] +
        0.2 * hybrid["clarity"], 3
    )

    return hybrid
