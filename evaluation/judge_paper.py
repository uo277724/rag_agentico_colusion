# evaluation/judge_paper.py
"""
LLM-as-a-Judge Agent — basado en el artículo de métricas de evaluación de modelos generativos.
Evalúa la respuesta del RAG según criterios de claridad, relevancia, insightfulness y contextualización.
Devuelve una puntuación detallada y un resumen de evaluación.
"""

import json
from openai import OpenAI


def evaluate_with_criteria(question: str, context: str, answer: str, model_name: str = "gpt-4o-mini"):
    """
    Evalúa la respuesta generada por el RAG siguiendo criterios definidos en el artículo.
    
    Parámetros:
        question (str): Pregunta del usuario.
        context (str): Contexto recuperado por el RAG.
        answer (str): Respuesta generada por el sistema.
        model_name (str): Modelo de evaluación LLM.

    Retorna:
        dict: Resultados de evaluación con puntuaciones y feedback.
    """

    client = OpenAI()

    # --- Prompt del juez ---
    prompt = f"""
Eres un juez técnico experto encargado de evaluar la calidad de respuestas generadas por un sistema RAG 
basado en documentación industrial. Debes asignar puntuaciones de 0 a 4 según los criterios siguientes:

1 CLARITY (Claridad)
- 0: Inadecuado – Confuso y ambiguo.
- 1: Necesita mejora – Algunas partes comprensibles, pero difícil de seguir.
- 2: Satisfactorio – Generalmente claro, con pequeños problemas.
- 3: Proficiente – Muy claro y bien estructurado.
- 4: Excelente – Perfectamente claro, preciso y técnico.

2 RELEVANCE (Relevancia)
- 0: Inadecuado – No responde a la pregunta.
- 1: Necesita mejora – Parcialmente relacionado, pero con divagaciones.
- 2: Satisfactorio – Mayormente relevante.
- 3: Proficiente – Altamente enfocado en la pregunta.
- 4: Excelente – Completamente relevante y centrado.

3 INSIGHTFULNESS (Perspicacia / Profundidad)
- 0: Inadecuado – Descriptivo o superficial.
- 1: Necesita mejora – Aporta poco valor nuevo.
- 2: Satisfactorio – Algunos elementos valiosos.
- 3: Proficiente – Contiene ideas útiles o inferencias claras.
- 4: Excelente – Aporta comprensión profunda o inferencias significativas.

4 CONTEXTUALIZATION (Contextualización)
- 0: Inadecuado – Sin conexión con el contexto o las implicaciones.
- 1: Necesita mejora – Mínima conexión contextual.
- 2: Satisfactorio – Algo de contextualización presente.
- 3: Proficiente – Buena conexión con el contexto o implicaciones.
- 4: Excelente – Plena contextualización técnica y operativa.

---

Devuelve **únicamente un objeto JSON** con esta estructura exacta:
{{
  "Clarity": <int>,
  "Relevance": <int>,
  "Insightfulness": <int>,
  "Contextualization": <int>,
  "Overall_score": <float>,
  "Feedback": "<comentario breve sobre la respuesta>"
}}

PREGUNTA:
{question}

CONTEXTO (extracto relevante):
{context[:1000]}

RESPUESTA GENERADA:
{answer}
"""

    try:
        # Se solicita al modelo que devuelva un JSON válido
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Eres un juez experto en documentación técnica industrial."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500
        )
        text = response.choices[0].message.content.strip()
        result = json.loads(text)

    except Exception as e:
        print(f"[JUDGE_PAPER ERROR] {e}")
        try:
            # Segundo intento (por si devuelve texto plano)
            text = response.choices[0].message.content.strip()
            json_str = text[text.find("{"):text.rfind("}")+1]
            result = json.loads(json_str)
        except Exception:
            result = {
                "Clarity": 0,
                "Relevance": 0,
                "Insightfulness": 0,
                "Contextualization": 0,
                "Overall_score": 0.0,
                "Feedback": "No se pudo evaluar correctamente la respuesta."
            }

    # Cálculo del promedio general si no viene incluido
    if "Overall_score" not in result or result["Overall_score"] == 0:
        result["Overall_score"] = round((
            result.get("Clarity", 0) +
            result.get("Relevance", 0) +
            result.get("Insightfulness", 0) +
            result.get("Contextualization", 0)
        ) / 4, 2)

    return result
