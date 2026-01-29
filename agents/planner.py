# agents/screening_planner.py

import json
from typing import Dict, Any, List
from openai import OpenAI

from agents.consolidation.bid_consolidator import BidConsolidationAgent


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
    Planner agéntico para:
    - RAG documental
    - Screening numérico de licitaciones
    """

    def __init__(self, tool_manager, calculation_agents: Dict[str, Any]):
        self.client = OpenAI()
        self.tool_manager = tool_manager
        self.calculation_agents = calculation_agents
        self.consolidator = BidConsolidationAgent()

    # --------------------------------------------------
    # Prompt de detección (clasificación pura)
    # --------------------------------------------------
    def _system_prompt(self) -> str:
        return """
You are a STRICT SCREENING INTENT CLASSIFIER.

TASK:
1. Determine whether the user requests:
   - DOCUMENTATION-ONLY information
   - NUMERICAL SCREENING of bids

2. If screening is requested, map to SUPPORTED METRICS ONLY.

SUPPORTED METRICS:
- cv, spd, diffp, rd, kurt, skew, kstest

RULES:
- Return ONLY a JSON object.
- Do NOT invent metrics.
- Do NOT compute anything.
- If screening intent exists but no supported metrics apply,
  return intent="screening" and metrics=[].
"""

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(self, query: str) -> Dict[str, Any]:
        print("\n[PLANNER] ======================")
        print("[PLANNER] Query:", query)

        # ----------------------------------------------
        # STEP 1: INTENT + METRIC DETECTION
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

Return format:
{{
  "intent": "rag" | "screening",
  "metrics": ["cv", "rd", ...]
}}
"""
                }
            ],
            max_tokens=200
        )

        raw_output = completion.choices[0].message.content
        print("[PLANNER] Raw classifier output:", raw_output)

        try:
            parsed = json.loads(raw_output)
            intent = parsed.get("intent")
            requested_metrics = [m.lower() for m in parsed.get("metrics", [])]
        except Exception as e:
            print("[PLANNER] ❌ JSON parse error:", str(e))
            return {
                "ok": False,
                "error": "Failed to parse planner output",
                "raw_output": raw_output
            }

        print("[PLANNER] Parsed intent:", intent)
        print("[PLANNER] Parsed metrics:", requested_metrics)

        # Normalización defensiva del intent
        if intent not in ("rag", "screening"):
            print("[PLANNER] ⚠️ Unknown intent, defaulting to 'rag'")
            intent = "rag"

        # ----------------------------------------------
        # STEP 2: DOCUMENTAL RAG
        # ----------------------------------------------
        if intent == "rag":
            print("[PLANNER] Entering RAG mode")

            rag = self.tool_manager.execute(
                "rag_query",
                {"query": query}
            )

            print("[PLANNER] RAG tool ok:", rag.get("ok"))

            if not rag.get("ok"):
                return {
                    "ok": False,
                    "error": "RAG query failed",
                    "details": rag
                }

            result = rag.get("result", {})
            print("[PLANNER] RAG completed successfully")

            return {
                "ok": True,
                "mode": "rag",
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "evaluation": result.get("evaluation"),
                "metrics": result.get("metrics"),
            }

        # ----------------------------------------------
        # STEP 3: SCREENING VALIDATION
        # ----------------------------------------------
        print("[PLANNER] Entering SCREENING mode")

        if not requested_metrics:
            print("[PLANNER] ❌ Screening intent but no metrics detected")
            return {
                "ok": False,
                "mode": "screening",
                "error": "Screening intent detected but no supported metrics identified",
                "supported_metrics": list(SCREENING_METRICS.keys()),
            }

        for m in requested_metrics:
            if m not in SCREENING_METRICS:
                print(f"[PLANNER] ❌ Unsupported metric: {m}")
                return {
                    "ok": False,
                    "error": f"Unsupported screening metric '{m}'"
                }

        # ----------------------------------------------
        # STEP 4: EXTRACT BID CANDIDATES (RAG)
        # ----------------------------------------------
        print("[PLANNER] Starting bid extraction")

        extraction = self.tool_manager.execute(
            "rag_extract_bids",
            {"query": query}
        )

        print("[PLANNER] Extraction ok:", extraction.get("ok"))

        if not extraction.get("ok"):
            return {
                "ok": False,
                "mode": "screening",
                "error": "Bid extraction failed",
                "details": extraction
            }

        payload = extraction.get("result", {})
        print("[PLANNER] Extraction payload keys:", payload.keys())
        print("[PLANNER] Raw extracted bids:", payload.get("bids"))

        # ----------------------------------------------
        # STEP 5: CONSOLIDATE BIDS (CRITICAL)
        # ----------------------------------------------
        print("[PLANNER] Starting bid consolidation")

        try:
            consolidated = self.consolidator.consolidate(payload)
        except Exception as e:
            print("[PLANNER] ❌ Consolidation error:", str(e))
            return {
                "ok": False,
                "mode": "screening",
                "error": "Bid consolidation failed",
                "details": str(e)
            }

        print("[PLANNER] Consolidation result:", consolidated)

        final_bids = consolidated.get("final_bids", [])
        print("[PLANNER] Final bids:", final_bids)

        if not final_bids:
            print("[PLANNER] ❌ No valid bids after consolidation")
            return {
                "ok": False,
                "mode": "screening",
                "error": "No valid bids after consolidation"
            }

        # ----------------------------------------------
        # STEP 6: CARDINALITY CHECK
        # ----------------------------------------------
        for metric in requested_metrics:
            min_n = SCREENING_METRICS[metric]["min_n"]
            print(
                f"[PLANNER] Cardinality check for {metric}: "
                f"{len(final_bids)} bids (min required: {min_n})"
            )
            if len(final_bids) < min_n:
                return {
                    "ok": False,
                    "mode": "screening",
                    "error": f"Not enough bids for metric '{metric}'",
                    "required": min_n,
                    "provided": len(final_bids),
                }

        # ----------------------------------------------
        # STEP 7: CALCULATION AGENTS
        # ----------------------------------------------
        results = {}
        for metric in requested_metrics:
            print(f"[PLANNER] Executing calculation agent: {metric}")
            agent = self.calculation_agents.get(metric)

            if agent is None:
                return {
                    "ok": False,
                    "error": f"No calculation agent for '{metric}'"
                }

            results[metric] = agent.compute(final_bids)
            print(f"[PLANNER] Result {metric} =", results[metric])

        # ----------------------------------------------
        # STEP 8: QUALITATIVE EXPLANATION
        # ----------------------------------------------
        print("[PLANNER] Generating qualitative explanation")

        explanation_completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": """
You interpret FINAL screening indicators ONLY.

RULES:
- Do NOT compute or derive anything.
- Do NOT explain formulas.
- Do NOT infer additional facts.
- Keep a neutral, non-accusatory tone.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Screening results:
{json.dumps(results, indent=2)}

Number of bids: {len(final_bids)}

Explain what these indicators MAY suggest.
"""
                }
            ],
            max_tokens=250
        )

        explanation = explanation_completion.choices[0].message.content.strip()
        print("[PLANNER] Explanation generated")

        # ----------------------------------------------
        # STEP 9: FINAL RESPONSE
        # ----------------------------------------------
        print("[PLANNER] Screening completed successfully")

        return {
            "ok": True,
            "mode": "screening",
            "metrics": results,
            "explanation": explanation,
            "meta": {
                "n_bids": len(final_bids),
                "currency": consolidated.get("currency"),
                "confidence": consolidated.get("confidence"),
                "decisions": consolidated.get("decisions"),
                "source_docs": payload.get("source_docs"),
            }
        }

        # ----------------------------------------------
        # FALLBACK (SHOULD NEVER HAPPEN)
        # ----------------------------------------------
        print("[PLANNER] ❌ Reached unexpected end of planner.run")

        return {
            "ok": False,
            "error": "Planner reached an unexpected state",
            "debug": {
                "query": query,
                "intent": intent,
                "metrics": requested_metrics
            }
        }
