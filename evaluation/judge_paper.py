"""
LLM-as-a-Judge orientado a fidelidad, trazabilidad y seguridad analítica.
No evalúa creatividad ni insight.
"""

import json
from openai import OpenAI
from typing import Dict


def evaluate_with_criteria(
    question: str,
    context: str,
    answer: str,
    model_name: str = "gpt-4o-mini",
) -> Dict:
    client = OpenAI()

    prompt = f"""
You are a STRICT EVALUATION JUDGE for a documentation-based analytical RAG system.

CRITICAL EVALUATION RULES:
- Do NOT reward creativity, insight, or interpretation.
- Penalize any inference not explicitly supported by the context.
- Penalize any number, claim, or conclusion not present in the context.
- If the answer extrapolates beyond the context, mark it as an error.

Evaluate the answer using the following criteria (0–4):

1. CLARITY
- Is the answer clear, precise, and well structured?

2. RELEVANCE
- Does the answer directly address the question?

3. FAITHFULNESS
- Is the answer strictly grounded in the provided context?
- Are all claims and numbers present in the context?

4. TRACEABILITY
- Is it clear how the answer relates to the context?
- Could a reviewer trace statements back to the context?

Additionally, identify ISSUE TYPES if present:
- "textual_issue"
- "missing_evidence"
- "unsupported_inference"
- "analysis_error"
- "wrong_calculation"

Return ONLY a valid JSON object with EXACTLY this structure:
{{
  "Clarity": <int>,
  "Relevance": <int>,
  "Faithfulness": <int>,
  "Traceability": <int>,
  "Overall_score": <float>,
  "Issue_types": [<string>, ...],
  "Feedback": "<brief explanation>"
}}

QUESTION:
{question}

CONTEXT (excerpt):
{context[:1000]}

ANSWER:
{answer}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a conservative technical evaluator."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
        )

        result = json.loads(response.choices[0].message.content.strip())

    except Exception as e:
        print(f"[JUDGE ERROR] {e}")
        result = {
            "Clarity": 0,
            "Relevance": 0,
            "Faithfulness": 0,
            "Traceability": 0,
            "Overall_score": 0.0,
            "Issue_types": ["evaluation_error"],
            "Feedback": "The answer could not be evaluated reliably.",
        }

    # Calcular score si falta
    if not result.get("Overall_score"):
        result["Overall_score"] = round(
            (
                result.get("Clarity", 0)
                + result.get("Relevance", 0)
                + result.get("Faithfulness", 0)
                + result.get("Traceability", 0)
            )
            / 4,
            2,
        )

    # Normalizar Issue_types
    if "Issue_types" not in result or not isinstance(result["Issue_types"], list):
        result["Issue_types"] = []

    return result
