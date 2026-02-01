"""
Refinador de respuestas RAG orientado exclusivamente a calidad textual.
Nunca corrige razonamiento, extracción ni cálculos.
"""

import time
from openai import OpenAI
from typing import Dict


class ResponseRefiner:
    """
    Refina respuestas solo cuando los problemas detectados
    son de claridad, redacción o estructura.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        score_threshold: float = 2.5,
    ):
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.client = OpenAI()

    # --------------------------------------------------
    # Decisión de refinamiento
    # --------------------------------------------------
    def needs_refinement(self, judge_result: Dict) -> bool:
        """
        El refinamiento solo es posible si:
        - la puntuación es baja
        - y el juez NO indica errores analíticos o de contenido
        """

        score = judge_result.get("Overall_score", 0)
        issues = judge_result.get("Issue_types", [])

        # Si hay errores de contenido o razonamiento, NO refinar
        blocked = {"analysis_error", "missing_evidence", "wrong_calculation"}
        if any(i in blocked for i in issues):
            return False

        return score < self.score_threshold

    # --------------------------------------------------
    # Refinamiento textual
    # --------------------------------------------------
    def refine(
        self,
        question: str,
        answer: str,
        judge_result: Dict,
    ) -> Dict:
        """
        Refina exclusivamente la redacción del answer.
        Nunca añade información ni corrige contenido.
        """

        if not self.needs_refinement(judge_result):
            return {
                "refined_answer": answer,
                "status": "unchanged",
                "reason": "No procede refinamiento textual",
            }

        feedback = judge_result.get("Feedback", "Sin observaciones específicas")

        system_prompt = (
            "You are a STRICT TEXTUAL REFINER.\n\n"
            "ROLE:\n"
            "- You ONLY improve clarity, wording, and structure.\n"
            "- You DO NOT correct facts, reasoning, or calculations.\n\n"
            "FORBIDDEN:\n"
            "- Adding new information.\n"
            "- Inferring missing facts.\n"
            "- Modifying numbers or conclusions.\n"
            "- Explaining methods or reasoning.\n\n"
            "RULE:\n"
            "- If improving the answer would require any reasoning or inference,\n"
            "  return the original answer unchanged.\n"
        )

        user_prompt = f"""
        [QUESTION]
        {question}

        [ORIGINAL ANSWER]
        {answer}

        [JUDGE FEEDBACK]
        {feedback}

        Task:
        - Improve wording and clarity ONLY if possible without changing meaning.
        - Do not add, remove, or infer information.
        - Return ONLY the refined answer text.
        - If no safe improvement is possible, return the original answer verbatim.
        """

        try:
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=800,
            )

            refined = response.choices[0].message.content.strip()
            elapsed = round(time.time() - start, 2)

            return {
                "refined_answer": refined,
                "status": "refined",
                "reason": f"Refinamiento textual aplicado (score {judge_result.get('Overall_score')}/4)",
                "elapsed_time_sec": elapsed,
            }

        except Exception as e:
            return {
                "refined_answer": answer,
                "status": "error",
                "reason": f"Error durante refinamiento: {str(e)}",
            }
