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
    # Helper: generate explanation when calculation is not possible
    # --------------------------------------------------
    def _generate_unfeasible_explanation(
        self,
        query: str,
        reason: str,
        details: Dict[str, Any],
    ) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": """
You explain why a numerical screening indicator
cannot be computed.

RULES:
- Do NOT compute anything.
- Do NOT explain formulas.
- Do NOT invent data.
- Base the explanation ONLY on the provided facts.
- Use clear, professional, neutral language.
"""
                },
                {
                    "role": "user",
                    "content": f"""
User question:
\"\"\"{query}\"\"\" 

Reason why calculation is not possible:
{reason}

Details:
{json.dumps(details, indent=2)}

Explain this to the user in a clear way.
"""
                }
            ],
            max_tokens=200
        )

        return completion.choices[0].message.content.strip()

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
        except Exception:
            return {
                "ok": False,
                "error": "Failed to parse planner output",
                "raw_output": raw_output
            }

        print("[PLANNER] Parsed intent:", intent)
        print("[PLANNER] Parsed metrics:", requested_metrics)

        if intent not in ("rag", "screening"):
            intent = "rag"

        # ----------------------------------------------
        # STEP 2: DOCUMENTAL RAG
        # ----------------------------------------------
        if intent == "rag":
            rag = self.tool_manager.execute("rag_query", {"query": query})

            if not rag.get("ok"):
                return {
                    "ok": False,
                    "error": "RAG query failed",
                    "details": rag
                }

            result = rag.get("result", {})
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
            explanation = self._generate_unfeasible_explanation(
                query=query,
                reason="No screening metrics were identified in the request.",
                details={"supported_metrics": list(SCREENING_METRICS.keys())},
            )
            return {
                "ok": True,
                "mode": "screening",
                "status": "no_metrics",
                "answer": explanation,
            }

        for m in requested_metrics:
            if m not in SCREENING_METRICS:
                return {
                    "ok": False,
                    "error": f"Unsupported screening metric '{m}'"
                }

        # ----------------------------------------------
        # STEP 4: EXTRACT BID CANDIDATES (RAG)
        # ----------------------------------------------
        extraction = self.tool_manager.execute(
            "rag_extract_bids",
            {"query": query}
        )

        if not extraction.get("ok"):
            return {
                "ok": False,
                "mode": "screening",
                "error": "Bid extraction failed",
                "details": extraction
            }

        payload = extraction.get("result", {})
        source_refs = payload.get("source_refs", [])

        # ----------------------------------------------
        # STEP 5: CONSOLIDATE BIDS
        # ----------------------------------------------
        consolidated = self.consolidator.consolidate(payload)

        result = consolidated.get("result", {})
        decisions = consolidated.get("decisions", [])
        data = consolidated.get("data", {})

        final_bids = result.get("final_bids", [])


        if not final_bids:
            explanation = self._generate_unfeasible_explanation(
                query=query,
                reason="No valid economic bids were identified after consolidation.",
                details={},
            )
            return {
                "ok": True,
                "mode": "screening",
                "status": "no_valid_bids",
                "answer": explanation,
            }

        # ----------------------------------------------
        # STEP 6: CARDINALITY CHECK
        # ----------------------------------------------
        for metric in requested_metrics:
            min_n = SCREENING_METRICS[metric]["min_n"]

            if len(final_bids) < min_n:
                explanation = self._generate_unfeasible_explanation(
                    query=query,
                    reason="Insufficient number of valid bids.",
                    details={
                        "metric": metric,
                        "n_bids": len(final_bids),
                        "min_required": min_n,
                    },
                )
                return {
                    "ok": True,
                    "mode": "screening",
                    "status": "insufficient_bids",
                    "answer": explanation,
                    "meta": {
                        "metric": metric,
                        "n_bids": len(final_bids),
                        "min_required": min_n,
                        "currency": result.get("currency"),
                        "decisions": decisions,
                        "data": data,
                    }
                }

        # ----------------------------------------------
        # STEP 7: CALCULATION AGENTS (UNCHANGED)
        # ----------------------------------------------
        results = {}
        for metric in requested_metrics:
            agent = self.calculation_agents.get(metric)
            results[metric] = agent.compute(final_bids)

        # ----------------------------------------------
        # STEP 8: QUALITATIVE EXPLANATION (UNCHANGED)
        # ----------------------------------------------
        explanation_completion = self.client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": """
                You explain the outcome of a public tender screening analysis.

                Your response MUST:
                - Explicitly mention WHERE the bid data was found.
                - Cite document name(s) and page number(s) when available.
                - Refer to tables or sections if the semantic_type suggests it
                (e.g. 'tabla_economica', 'oferta_economica').

                STRUCTURE:
                1. First paragraph:
                - Explicitly state each requested indicator and its numeric value.

                2. Data origin:
                - Specify the document(s), page(s), and section/table where the bids appear.

                3. Then explain:
                - How many bids were considered
                - How bids were selected and consolidated
                - How reliable the data appears
                - What the indicators MAY suggest

                RULES:
                - Do NOT compute or derive new values
                - Do NOT invent sources or pages
                - Use ONLY the provided references
                - If a page is unknown, say so explicitly
                - Neutral, professional tone
                """
                },
            {
                "role": "user",
                "content": f"""
                Requested screening indicators (already computed):
                {json.dumps(results, indent=2)}

                Number of valid bids considered: {len(final_bids)}
                Currency: {result.get("currency")}

                Bid selection and consolidation decisions:
                {json.dumps(decisions, indent=2)}

                Data origin references for extracted bids.
                Each item includes:
                - source: document filename
                - page: page number (if available)
                - semantic_type: section type (e.g. table, economic offer, annex)

                References:
                {json.dumps(source_refs, indent=2)}

                Overall data confidence: {result.get("confidence")}

                Explain the results following the required structure.
                """
            }
        ],
        max_tokens=300
    )

        explanation = explanation_completion.choices[0].message.content.strip()

        # ----------------------------------------------
        # STEP 9: FINAL RESPONSE (UNCHANGED)
        # ----------------------------------------------
        return {
            "ok": True,
            "mode": "screening",
            "metrics": results,
            "explanation": explanation,
            "meta": {
                "n_bids": len(final_bids),
                "currency": result.get("currency"),
                "confidence": result.get("confidence"),
                "decisions": decisions,
                "data": data,
            }
        }

