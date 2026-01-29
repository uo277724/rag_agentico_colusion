# agents/tools/rag_extract_bids.py

from typing import Dict, Any, List, Union
from openai import OpenAI
from retrieval.retriever import Retriever
import json


class RAGExtractBidsTool:
    """
    RAG tool for extracting bid hypotheses from tender documentation.
    """

    def __init__(self, embedder, vectorstore):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.client = OpenAI()

    def __call__(self, query: str) -> Dict[str, Any]:

        print("\n[RAG_EXTRACT_BIDS] ======================")
        print("[RAG_EXTRACT_BIDS] Query:", query)

        # --------------------------------------------------
        # STEP 1: Retrieve evidence (text-based)
        # --------------------------------------------------
        retriever = Retriever(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )

        raw_results = retriever.retrieve(query)

        if not raw_results:
            print("[RAG_EXTRACT_BIDS] No evidence retrieved")
            return {"error": "No relevant documentation found"}

        print(f"[RAG_EXTRACT_BIDS] Raw units retrieved: {len(raw_results)}")

        # --------------------------------------------------
        # STEP 2: Normalize evidence to text chunks
        # --------------------------------------------------
        context_chunks: List[str] = []

        for r in raw_results:
            if isinstance(r, str):
                context_chunks.append(r)

            elif isinstance(r, list):
                for item in r:
                    if isinstance(item, str):
                        context_chunks.append(item)

            elif isinstance(r, dict):
                content = r.get("content")
                if isinstance(content, str):
                    context_chunks.append(content)

        context_chunks = [c.strip() for c in context_chunks if c.strip()]
        context = "\n---\n".join(context_chunks)

        print(f"[RAG_EXTRACT_BIDS] Context length: {len(context)} chars")

        if not context:
            print("[RAG_EXTRACT_BIDS] Empty context after normalization")
            return {"error": "Empty context after retrieval"}

        # --------------------------------------------------
        # STEP 3: Structured extraction with LLM
        # --------------------------------------------------
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a STRICT BID EXTRACTION AGENT.

TASK:
- Identify ALL distinct economic bids explicitly stated in the text.
- A bid is a price offered by a bidder for the contract.
- Ignore budgets, guarantees, penalties, thresholds, estimates, or references.
- Distinguish between different bidders if possible.
- Distinguish tax-inclusive vs tax-exclusive amounts IF explicitly stated.
- Do NOT infer or calculate missing values.
- Do NOT merge amounts.
- If information is ambiguous, include it with low confidence.

OUTPUT:
Return ONLY a valid JSON array.

Each element must have:
- bidder (string or null)
- amount (number)
- currency (string or null)
- tax_included (true | false | null)
- source_excerpt (string)
- confidence (number between 0 and 1)
"""
                },
                {
                    "role": "user",
                    "content": f"""
DOCUMENT TEXT:
\"\"\"{context}\"\"\" 

Extract bid candidates.
"""
                }
            ],
            max_tokens=1200
        )

        raw_output = completion.choices[0].message.content.strip()

        print("[RAG_EXTRACT_BIDS] Raw LLM output:")
        print(raw_output)

        clean_output = _strip_markdown_fences(raw_output)

        try:
            bid_candidates = json.loads(clean_output)
        except Exception as e:
            print("[RAG_EXTRACT_BIDS] JSON parse error:", e)
            return {
                "error": "Failed to parse bid extraction output",
                "raw_output": raw_output
            }

        if not bid_candidates:
            return {"error": "No bid candidates found"}

        # --------------------------------------------------
        # STEP 4: Minimal normalization
        # --------------------------------------------------
        normalized = []

        for b in bid_candidates:
            try:
                amount = float(b.get("amount"))
            except Exception:
                continue

            normalized.append({
                "bidder": b.get("bidder"),
                "amount": amount,
                "currency": b.get("currency"),
                "tax_included": b.get("tax_included"),
                "source_excerpt": (b.get("source_excerpt") or "")[:500],
                "confidence": float(b.get("confidence", 0.5)),
            })

        if not normalized:
            return {"error": "No valid bid amounts extracted"}

        avg_confidence = round(
            sum(b["confidence"] for b in normalized) / len(normalized), 2
        )

        print("[RAG_EXTRACT_BIDS] Average confidence:", avg_confidence)
        print("[RAG_EXTRACT_BIDS] ======================\n")

        return {
            "bids": normalized,
            "confidence": avg_confidence,
        }
    
def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text
