# evaluation/refiner.py
"""
Módulo de refinamiento de respuestas RAG basado en feedback del juez (LLM-as-a-Judge).
Si la puntuación general es baja, reformula o amplía la respuesta original para mejorarla.
"""

import json
from openai import OpenAI
import time


class ResponseRefiner:
    """
    Clase encargada de revisar y mejorar respuestas generadas por el sistema RAG
    cuando el módulo de evaluación (judge_paper) detecta calidad insuficiente.
    """

    def __init__(self, model_name="gpt-4o-mini", threshold=2.5):
        """
        Parámetros:
            model_name (str): Modelo a usar para el refinamiento.
            threshold (float): Umbral mínimo de puntuación media (0–4) para activar el refinamiento.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.client = OpenAI()

    def needs_refinement(self, judge_result):
        """
        Comprueba si la respuesta debe refinarse según la puntuación del juez.
        """
        return judge_result.get("Overall_score", 0) < self.threshold

    def refine(self, question, context, answer, judge_result):
        """
        Mejora la respuesta utilizando el feedback del juez.
        Si no se requiere refinamiento, devuelve la respuesta original.
        """
        if not self.needs_refinement(judge_result):
            return {"refined_answer": answer, "status": "unchanged", "reason": "Puntuación suficiente"}

        feedback = judge_result.get("Feedback", "Sin observaciones específicas")

        system_prompt = (
            "You are a DOCUMENTATION-ONLY RAG RESPONSE REFINER.\n\n"
            "CRITICAL ROLE DEFINITION:\n"
            "- You are NOT an analyst.\n"
            "- You are NOT a calculator.\n"
            "- You are NOT a reasoning engine.\n"
            "- You are a textual editor working on top of an existing answer.\n\n"
            "STRICT RULES (NO EXCEPTIONS):\n"
            "- You must NOT perform calculations of any kind.\n"
            "- You must NOT derive values, averages, variances, ratios, or statistics.\n"
            "- You must NOT infer information that is not explicitly written in the context.\n"
            "- You must NOT explain mathematical procedures.\n"
            "- You must NOT combine numbers to obtain new numbers.\n\n"
            "ALLOWED ACTIONS ONLY:\n"
            "- Improve wording, clarity, or structure of the EXISTING answer.\n"
            "- Rephrase sentences without changing meaning.\n"
            "- Remove redundancies.\n\n"
            "ABSOLUTE CONSTRAINT:\n"
            "- If improving the answer would require reasoning, calculation, or inference,\n"
            "  you MUST return the original answer unchanged.\n"
        )

        user_prompt = f"""
            [QUESTION]
            {question}

            [CONTEXT]
            {context[:1500]}

            [ORIGINAL ANSWER]
            {answer}

            [JUDGE FEEDBACK]
            {feedback}

            Instructions:
            - Improve the answer based solely on the context and judge feedback.
            - Do not add information outside the context.
            - Return only the revised version of the answer without explanations.
            - Do not invent anything; do not include anything not present in the information. 
            - If you don't find it, only say that you haven't found it and don't add any more text.
            """


        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000
            )
            refined = response.choices[0].message.content.strip()
            elapsed = round(time.time() - start_time, 2)

            result = {
                "refined_answer": refined,
                "status": "refined",
                "reason": f"Mejorada automáticamente por baja puntuación ({judge_result['Overall_score']}/4)",
                "elapsed_time_sec": elapsed
            }

        except Exception as e:
            result = {
                "refined_answer": answer,
                "status": "error",
                "reason": f"Error en refinamiento: {str(e)}"
            }

        return result
