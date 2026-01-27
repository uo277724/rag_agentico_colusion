# agents/screening_planner.py

import json
from typing import Dict, Any, List
from openai import OpenAI


# --------------------------------------------------
# Registro de métricas de screening
# --------------------------------------------------
SCREENING_METRICS = {
    "cv": {"min_n": 2},
    "spd": {"min_n": 2},
    "diffp": {"min_n": 2},
    "rd": {"min_n": 3},
    "kurt": {"min_n": 4},
    "skew": {"min_n": 3},
    "kstest": {"min_n": 3},
}


class ScreeningPlannerAgent:
    """
    Unified planner for:
    - RAG documental narrativo
    - Screening numérico de licitaciones

    Responsibilities:
    - Detect screening metrics from natural language
    - Route to RAG or Screening pipeline
    - Execute tools explicitly
    - Validate numeric inputs
    - Dispatch deterministic calculation agents
    - Generate human-readable explanations
    - Return structured, auditable output

    This planner:
    - NEVER computes metrics itself
    - NEVER invents numeric values
    - NEVER lets the LLM choose tools
    """

    def __init__(self, tool_manager, calculation_agents: Dict[str, Any]):
        self.client = OpenAI()
        self.tool_manager = tool_manager
        self.calculation_agents = calculation_agents

    # --------------------------------------------------
    # System prompt (metric detection only)
    # --------------------------------------------------
    def _system_prompt(self) -> str:
        return """
You are a metric classification agent for a public tender screening system.

Your task is to interpret the user's request and map it to the supported
screening metrics IF POSSIBLE.

SUPPORTED METRICS:
- cv      (coefficient of variation, dispersion of bids)
- spd     (spread between highest and lowest bid)
- diffp   (difference between two lowest bids)
- rd      (relative distance between lowest bids)
- kurt    (kurtosis of bid distribution)
- skew    (skewness of bid distribution)
- kstest  (uniformity test of bids)

RULES:
- You MUST return ONLY valid metric identifiers from the supported list.
- You MAY map natural language expressions to the closest supported metric.
- If the requested concept CANNOT be mapped to a supported metric, DO NOT invent one.
- If no supported metrics apply, return an empty JSON array [].
- Do NOT compute anything.
- Do NOT explain your reasoning.
"""

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(self, query: str) -> Dict[str, Any]:
        print("\n==============================")
        print("DEBUG: ScreeningPlanner.run")
        print("DEBUG: Query:", query)
        print("==============================")

        # ----------------------------------------------
        # STEP 1: detect requested screening metrics
        # ----------------------------------------------
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": f"""
User query:
\"\"\"{query}\"\"\"

Allowed screening metrics:
{list(SCREENING_METRICS.keys())}

Return ONLY a JSON array of metric identifiers.
Example: ["cv", "rd"]
If none are requested, return [].
"""
                }
            ],
            max_tokens=200
        )

        try:
            requested_metrics: List[str] = json.loads(
                completion.choices[0].message.content
            )
        except Exception:
            return {
                "ok": False,
                "error": "Failed to parse screening metrics from LLM output"
            }

        # Normalización defensiva
        requested_metrics = [m.lower() for m in requested_metrics]

        print("DEBUG: Requested metrics:", requested_metrics)

        # ----------------------------------------------
        # STEP 2: RAG DOCUMENTAL (no metrics requested)
        # ----------------------------------------------
        if not requested_metrics:
            rag = self.tool_manager.execute(
                "rag_query",
                {"query": query}
            )

            if not rag.get("ok"):
                return {
                    "ok": False,
                    "error": "RAG query failed",
                    "details": rag
                }

            result = rag.get("result", {})

            return {
                "ok": True,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", [])
            }

        # ----------------------------------------------
        # STEP 3: validate requested metrics
        # ----------------------------------------------
        for metric in requested_metrics:
            if metric not in SCREENING_METRICS:
                return {
                    "ok": False,
                    "error": f"Unknown screening metric '{metric}'"
                }

        # ----------------------------------------------
        # STEP 4: extract bids via RAG
        # ----------------------------------------------
        extraction = self.tool_manager.execute(
            "rag_extract_bids",
            {"query": query}
        )

        if not extraction.get("ok"):
            return {
                "ok": False,
                "error": "Failed to extract bids",
                "details": extraction
            }

        payload = extraction.get("result", {})

        bids = payload.get("bids")
        if not isinstance(bids, list) or not bids:
            return {
                "ok": False,
                "error": "No bids found in documentation"
            }

        print("DEBUG: Extracted bids:", bids)

        # ----------------------------------------------
        # STEP 5: validate cardinality
        # ----------------------------------------------
        for metric in requested_metrics:
            min_n = SCREENING_METRICS[metric]["min_n"]
            if len(bids) < min_n:
                return {
                    "ok": False,
                    "error": f"Not enough bids for metric '{metric}'",
                    "required": min_n,
                    "provided": len(bids)
                }

        # ----------------------------------------------
        # STEP 6: execute calculation agents
        # ----------------------------------------------
        results = {}

        for metric in requested_metrics:
            agent = self.calculation_agents.get(metric)
            if agent is None:
                return {
                    "ok": False,
                    "error": f"No calculation agent registered for '{metric}'"
                }

            print(f"DEBUG: Executing calculation agent '{metric}'")

            try:
                results[metric] = agent.compute(bids)
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Calculation failed for '{metric}'",
                    "details": str(e)
                }

        # ----------------------------------------------
        # STEP 7: natural language explanation (NEW)
        # ----------------------------------------------
        explanation_completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": """
        You are an explanatory assistant for PUBLIC TENDER SCREENING RESULTS.

        ROLE DEFINITION:
        - You interpret ALREADY COMPUTED screening indicators.
        - You do NOT calculate, derive, or reconstruct metrics.
        - You do NOT explain how indicators are computed.
        - You do NOT introduce intermediate statistics (mean, std, variance, etc.).
        - You do NOT infer numeric values not explicitly provided.

        STRICT RULES:
        - You may ONLY refer to the metric values explicitly provided.
        - You may ONLY interpret their qualitative meaning.
        - You must NOT restate formulas or computation steps.
        - You must NOT invent or assume additional data.
        - You must keep a neutral, non-accusatory tone.

        If interpretation is not possible without additional data,
        explicitly say that the indicator alone is insufficient.
        """
                },
                {
                    "role": "user",
                    "content": f"""
        Screening results (final values only):
        {json.dumps(results, indent=2)}

        Number of bids: {len(bids)}

        Explain what these results MAY indicate in the context of a public tender.
        Do NOT explain how the indicators are calculated.
        """
                }
            ],
            max_tokens=250
        )

        explanation_text = (
            explanation_completion.choices[0].message.content.strip()
        )

        # ----------------------------------------------
        # STEP 8: final structured response
        # ----------------------------------------------
        return {
            "ok": True,
            "metrics": results,
            "explanation": explanation_text,
            "meta": {
                "n_bids": len(bids),
                "currency": payload.get("currency"),
                "source_docs": payload.get("source_docs"),
                "confidence": payload.get("confidence")
            }
        }
