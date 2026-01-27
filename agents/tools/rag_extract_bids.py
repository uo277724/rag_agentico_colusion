# agents/tools/rag/extract_bids.py

import re
from typing import Dict, Any, List
from openai import OpenAI

from retrieval.retriever import Retriever


class RAGExtractBidsTool:
    """
    RAG tool for extracting bid values from tender documentation.
    """

    def __init__(self, embedder, vectorstore):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.client = OpenAI()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _extract_numbers(self, text: str) -> List[float]:
        """
        Extract numeric values from text.
        Handles simple European formats.
        """
        text = text.replace(".", "").replace(",", ".")
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        return [float(n) for n in numbers]

    # --------------------------------------------------
    # Tool call
    # --------------------------------------------------
    def __call__(self, query: str) -> Dict[str, Any]:

        # --------------------------------------------------
        # STEP 1: Retrieve relevant context (REAL RAG)
        # --------------------------------------------------
        retriever = Retriever(self.embedder, self.vectorstore, top_k=10)
        results = retriever.retrieve(query)

        context = results.get("context", "").strip()
        sources = results.get("sources", [])

        if not context:
            return {
                "error": "No relevant documentation found"
            }

        # --------------------------------------------------
        # STEP 2: LLM extraction over real context
        # --------------------------------------------------
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an information extraction assistant.

TASK:
- Extract ONLY bid amounts explicitly mentioned in the text.
- Ignore budgets, guarantees, penalties or other amounts.
- Return ONLY text fragments that contain bid values.
- Do NOT compute anything.
"""
                },
                {
                    "role": "user",
                    "content": f"""
DOCUMENT TEXT:
\"\"\"{context}\"\"\"

Extract bid values.
"""
                }
            ],
            max_tokens=1000
        )

        extracted_text = completion.choices[0].message.content or ""

        # --------------------------------------------------
        # STEP 3: Parse numeric values
        # --------------------------------------------------
        bids = self._extract_numbers(extracted_text)

        if not bids:
            return {
                "error": "No bids found in documents"
            }

        # --------------------------------------------------
        # STEP 4: Currency detection
        # --------------------------------------------------
        currency = None
        if "€" in extracted_text or "EUR" in extracted_text:
            currency = "EUR"
        elif "$" in extracted_text or "USD" in extracted_text:
            currency = "USD"

        return {
            "bids": bids,
            "currency": currency,
            "source_docs": sources,
            "confidence": 0.8
        }
