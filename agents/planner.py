# agents/screening_planner.py

import json
from typing import Dict, Any, List
from openai import OpenAI

from agents.consolidation.bid_consolidator import BidConsolidationAgent
from agents.interpretation.screening_assessment import ScreeningAssessmentAgent
from memory.memory_store import MemoryStore
from agents.memory_resolver import MemoryResolverAgent

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

    def __init__(self, tool_manager, calculation_agents: Dict[str, Any], memory_store: MemoryStore):
        self.client = OpenAI()
        self.tool_manager = tool_manager
        self.calculation_agents = calculation_agents
        self.consolidator = BidConsolidationAgent()
        self.memory_store = memory_store
        self.memory_resolver = MemoryResolverAgent()
        self.interpretation_agent = ScreeningAssessmentAgent()

    # --------------------------------------------------
    # Prompt de detección (clasificación pura)
    # --------------------------------------------------
    def _system_prompt(self) -> str:
        return """
        You are a STRICT SCREENING INTENT CLASSIFIER.

        TASK:
        1. Determine whether the user requests:
        - DOCUMENTATION-ONLY information
        - NUMERICAL SCREENING of bids (explicit indicators requested)
        - GLOBAL SCREENING ASSESSMENT (general evaluation of possible collusion indicators)

        2. If screening is requested, map to SUPPORTED METRICS ONLY.

        SUPPORTED METRICS:
        - cv, spd, diffp, rd, kurt, skew, kstest

        A request is a GLOBAL SCREENING ASSESSMENT if:
        - The user asks whether there are indications, signs, risks or patterns of collusion
        - The user does NOT request specific indicators or formulas
        - The question is evaluative or interpretative rather than computational

        RULES:
        - Return ONLY a JSON object.
        - Do NOT invent metrics.
        - Do NOT compute anything.
        - For GLOBAL SCREENING ASSESSMENT, metrics MAY be empty.
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
    
    def _reinterpret_query_with_memory(
        self,
        query: str,
        memory: Dict[str, Any],
    ) -> str:
        """
        Rewrites the user query to make it fully self-contained
        using conversation memory, without adding new facts.
        """

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": """
                You rewrite user follow-up questions to make them fully explicit and
                document-anchored.

                CRITICAL RULES:
                - You MUST preserve the original documentary subject (meeting, act, table, document).
                - You MUST preserve roles or entities already implied (e.g. president, members, attendees, vocales).
                - You MUST NOT generalize the question.
                - You MUST NOT remove references to the document or event if present in memory.
                - You MUST NOT simplify the question into a generic one.
                - Use ONLY the provided memory facts.
                - Do NOT add new information.
                - Do NOT answer the question.
                - Do NOT explain anything.

                GOAL:
                Rewrite the question so that it clearly refers to the same document,
                event, and section as the previous turn, making implicit references explicit.

                Return ONLY the rewritten question.
                """
                },
                {
                    "role": "user",
                    "content": f"""
                    Current question:
                    {query}

                    Conversation memory:
                    {json.dumps(memory, indent=2)}
                    """
                }
            ]
        )

        return completion.choices[0].message.content.strip()
    
    # --------------------------------------------------
    # Helper: render structured assessment into text
    def _render_assessment_text(
        self,
        interpretation: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> str:
        lines = []

        # ----------------------------------
        # Assessment level
        # ----------------------------------
        level = interpretation.get("assessment_level", "inconclusive")
        lines.append(f"**Assessment level:** {level}\n")

        # ----------------------------------
        # Summary
        # ----------------------------------
        summary = interpretation.get("summary")
        if summary:
            lines.append(summary + "\n")

        # ----------------------------------
        # Metric observations (WITH VALUES)
        # ----------------------------------
        obs = interpretation.get("metric_observations", {})
        if obs:
            lines.append("**Indicator observations:**\n")
            for metric, text in obs.items():
                value = metrics.get(metric, {}).get("value")
                if value is not None:
                    lines.append(f"- **{metric}** (value = {value:.4f}): {text}")
                else:
                    lines.append(f"- **{metric}**: {text}")

        # ----------------------------------
        # Limitations
        # ----------------------------------
        limitations = interpretation.get("limitations", [])
        if limitations:
            lines.append("\n**Limitations:**")
            for l in limitations:
                lines.append(f"- {l}")

        # ----------------------------------
        # Disclaimer
        # ----------------------------------
        disclaimer = interpretation.get("disclaimer")
        if disclaimer:
            lines.append("\n*" + disclaimer + "*")

        return "\n".join(lines)

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(self, query: str, conversation_id: str) -> Dict[str, Any]:
        print("\n[PLANNER] ======================")

        memory = self.memory_store.get_state(conversation_id)
        print("[PLANNER] Memory snapshot:", memory)

        print("[PLANNER] Query:", query)

        # ----------------------------------------------
        # MEMORY DEPENDENCY CHECK
        # ----------------------------------------------
        resolution = self.memory_resolver.needs_memory(query)

        print("[PLANNER] Memory resolution:", resolution)

        effective_query = query

        if resolution.get("needs_memory") and memory:
            effective_query = self._reinterpret_query_with_memory(
                query=query,
                memory=memory
            )
            print("[PLANNER] Rewritten query:", effective_query)

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
                    \"\"\"{effective_query}\"\"\" 

                    Return format:
                    {{
                    "intent": "rag" | "screening" | "screening_assessment",
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

        if intent not in ("rag", "screening", "screening_assessment"):
            intent = "rag"

        # ----------------------------------------------
        # STEP 2: DOCUMENTAL RAG
        # ----------------------------------------------
        if intent == "rag":
            rag = self.tool_manager.execute(
                "rag_query",
                {"query": effective_query}
            )

            if not rag.get("ok"):
                return {
                    "ok": False,
                    "error": "RAG query failed",
                    "details": rag
                }

            result = rag.get("result", {})

            self.memory_store.update_state(
                conversation_id,
                {
                    "last_mode": "rag",
                    "last_question": query,
                    "last_answer": result.get("answer"),
                    "last_sources": result.get("sources", []),
                }
            )

            return {
                "ok": True,
                "mode": "rag",
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "evaluation": result.get("evaluation"),
                "metrics": result.get("metrics"),
            }
        
        # ----------------------------------------------
        # STEP 2bis: GLOBAL SCREENING ASSESSMENT
        if intent == "screening_assessment":
            requested_metrics = list(SCREENING_METRICS.keys())

        # ----------------------------------------------
        # STEP 3: SCREENING VALIDATION
        # ----------------------------------------------
        print("[PLANNER] Entering SCREENING mode")

        if intent == "screening" and not requested_metrics:
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
            {"query": effective_query}
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

        print("[PLANNER] Computed screening metrics:")
        for k, v in results.items():
            print(f"  - {k}: {v}")

        print("[PLANNER] Computed metrics:", results)

        # ----------------------------------------------
        # STEP 8bis: INTERPRETATIVE ASSESSMENT
        # ----------------------------------------------
        if intent == "screening_assessment":
            interpretation = self.interpretation_agent.assess(
                metrics=results,
                context={
                    "n_bids": len(final_bids),
                    "confidence": result.get("confidence"),
                    "currency": result.get("currency"),
                }
            )

            print(interpretation)

            self.memory_store.update_state(
                conversation_id,
                {
                    "last_mode": "screening_assessment",
                    "last_metrics": list(results.keys()),
                    "n_bids": len(final_bids),
                    "confidence": result.get("confidence"),
                }
            )

            assessment_text = self._render_assessment_text(
                interpretation=interpretation,
                metrics=results
            )

            return {
                "ok": True,
                "mode": "screening_assessment",
                "metrics": results,
                "explanation": assessment_text,   # ← clave para la UI
                "assessment": interpretation,
                "meta": {
                    "n_bids": len(final_bids),
                    "currency": result.get("currency"),
                    "confidence": result.get("confidence"),
                }
            }

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

                - You MUST NOT assume whether amounts include or exclude VAT.
                - VAT handling must be described ONLY using the provided consolidation decisions.

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

        self.memory_store.update_state(
            conversation_id,
            {
                "last_mode": "screening",
                "last_metrics": list(results.keys()),
                "n_bids": len(final_bids),
                "currency": result.get("currency"),
                "confidence": result.get("confidence"),
                "last_decisions": decisions,
                "data_sources": source_refs,
            }
        )

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

