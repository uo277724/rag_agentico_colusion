# agents/interpretation/screening_assessment.py

import json
from typing import Dict, Any
from openai import OpenAI


class ScreeningAssessmentAgent:
    """
    Performs a qualitative, non-accusatory interpretation of
    pre-computed screening indicators for public procurement bids.

    This agent:
    - DOES NOT compute metrics
    - DOES NOT access documents
    - DOES NOT use conversational memory
    - DOES NOT infer illegality or wrongdoing

    It ONLY contextualizes numerical indicators to support
    early-stage analytical oversight.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def assess(
        self,
        metrics: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        metrics:
            Dictionary of screening indicators already computed.
            Example:
            {
              "cv": 0.04,
              "rd": 0.12,
              "spd": 0.01
            }

        context:
            Auxiliary non-numerical context, e.g.:
            {
              "n_bids": 6,
              "currency": "EUR",
              "non_computable_metrics": ["kstest"],
              "confidence": 0.9
            }

        Returns
        -------
        Structured qualitative assessment.
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.15,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an analytical assistant specialized in public procurement screening.

                    TASK:
                    Provide a qualitative interpretation of pre-computed screening indicators
                    used to identify potential anomalous bidding patterns.

                    CRITICAL RULES:
                    - You MUST NOT compute any metric.
                    - You MUST NOT introduce thresholds unless explicitly justified.
                    - You MUST NOT accuse, label, or imply illegal behavior.
                    - You MUST NOT infer intent or coordination.
                    - You MUST NOT reference external sources or documents.
                    - You MUST rely ONLY on the provided metrics and context.
                    - Use cautious, analytical, non-accusatory language.

                    OUTPUT FORMAT (VALID JSON ONLY):
                    {
                    "assessment_level": "no_indication | weak_signals | moderate_signals | inconclusive",
                    "summary": "short analytical overview",
                    "metric_observations": {
                        "<metric>": "interpretation"
                    },
                    "limitations": [
                        "limitation 1",
                        "limitation 2"
                    ],
                    "disclaimer": "standard disclaimer text"
                    }
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Computed screening indicators:
                    {json.dumps(metrics, indent=2)}

                    Contextual information:
                    {json.dumps(context, indent=2)}

                    Produce the qualitative assessment.
                    """
                }
            ]
        )

        raw = completion.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except Exception:
            # Fallback defensivo (nunca romper el sistema)
            return {
                "assessment_level": "inconclusive",
                "summary": "The available screening indicators could not be reliably interpreted.",
                "metric_observations": {},
                "limitations": [
                    "Automatic interpretation failed",
                    "Manual review recommended"
                ],
                "disclaimer": (
                    "This assessment does not imply the existence of collusion "
                    "or any form of unlawful behavior."
                ),
            }
