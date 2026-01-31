from typing import List, Dict, Any
from openai import OpenAI
import json
from collections import Counter


class LazySemanticTyper:
    """
    Anotador semántico perezoso.

    - Clasifica fragmentos ya recuperados (top-K).
    - NO filtra, NO reordena, NO descarta.
    - Añade metadata blanda: semantic_type + type_confidence.
    - Totalmente opcional y no bloqueante.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_chunks: int = 20,
        confidence_threshold: float = 0.0,
        debug: bool = False,
    ):
        self.client = OpenAI()
        self.model = model
        self.max_chunks = max_chunks
        self.confidence_threshold = confidence_threshold
        self.debug = debug

    # --------------------------------------------------
    # API pública
    # --------------------------------------------------
    def annotate(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return docs

        docs = docs[: self.max_chunks]

        if self.debug:
            print(f"[LAZY_TYPER] Typing {len(docs)} chunks")

        prompt = self._build_prompt(docs)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a STRICT DOCUMENT FUNCTION CLASSIFIER.\n"
                            "Classify each fragment into ONE functional type.\n"
                            "Do NOT extract data. Do NOT summarize. Do NOT invent.\n"
                            "Return ONLY valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
            )

            raw_output = response.choices[0].message.content.strip()

            if self.debug:
                print("[LAZY_TYPER] Raw model output:")
                print(raw_output)

            clean = self._strip_markdown_fences(raw_output)
            parsed = json.loads(clean)

        except Exception as e:
            if self.debug:
                print(f"[LAZY_TYPER] Failed, returning original docs: {e}")
            return docs

        # --------------------------------------------------
        # Anotar metadata sin alterar estructura
        # --------------------------------------------------
        for doc, ann in zip(docs, parsed):
            semantic_type = ann.get("type", "otro")
            confidence = float(ann.get("confidence", 0.0))

            meta = doc.setdefault("metadata", {})
            meta["semantic_type"] = semantic_type
            meta["type_confidence"] = confidence

            if self.debug:
                preview = doc.get("content", "")[:120].replace("\n", " ")
                print(
                    f"[LAZY_TYPER] → {semantic_type} "
                    f"(confidence={confidence:.2f}) | {preview}"
                )

        if self.debug:
            summary = Counter(
                d.get("metadata", {}).get("semantic_type", "otro")
                for d in docs
            )
            print("[LAZY_TYPER] Summary:", dict(summary))

        return docs

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------
    def _build_prompt(self, docs: List[Dict[str, Any]]) -> str:
        prompt = (
            "Classify each fragment into ONE functional type.\n\n"
            "Allowed types:\n"
            "- tabla_economica\n"
            "- oferta_economica\n"
            "- presupuesto_base\n"
            "- descripcion\n"
            "- anexo\n"
            "- otro\n\n"
            "For each fragment return an object with:\n"
            "{ \"type\": <string>, \"confidence\": <number between 0 and 1> }\n\n"
        )

        for i, d in enumerate(docs):
            text = d.get("content", "")[:800].replace("\n", " ")
            prompt += f"[{i}] {text}\n\n"

        prompt += (
            "Return a JSON array with one element per fragment, "
            "in the same order. No extra text."
        )

        return prompt
    

    # Función auxiliar para limpiar output
    def _strip_markdown_fences(self, text: str) -> str:
        """Remove markdown code fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return text
