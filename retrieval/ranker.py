import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from openai import OpenAI


class SemanticRanker:
    """
    Prioriza evidencia candidata a contener ofertas económicas reales
    usando similitud semántica en modo extracción.
    """

    def __init__(self, embedder, final_k: int = 15, verbose: bool = True):
        self.embedder = embedder
        self.final_k = final_k
        self.verbose = verbose

    def rerank(
        self,
        query: str,
        evidence: List[Dict],
    ) -> List[Dict]:
        if not evidence:
            return []

        if self.verbose:
            print("\n[RANKER] --- Ranking semántico (extract) ---")
            print(f"[RANKER] Evidencias iniciales: {len(evidence)}")

        # Construir embeddings en modo extract
        query_doc = {"content": query}
        query_emb = self.embedder.embed_documents(
            [query_doc], mode="extract"
        )[0]["embedding_extract"]

        docs_for_embedding = [
            {"content": e["content"]}
            for e in evidence
        ]

        embedded_docs = self.embedder.embed_documents(
            docs_for_embedding, mode="extract"
        )

        doc_embs = np.array(
            [d["embedding_extract"] for d in embedded_docs]
        )

        sims = cosine_similarity([query_emb], doc_embs)[0]

        # Combinar score semántico con confidence estructural
        scored = []
        for e, sim in zip(evidence, sims):
            conf = e["metadata"].get("confidence", 1.0)
            score = float(sim) * float(conf)
            scored.append((score, e))

        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        results = [e for _, e in ranked[: self.final_k]]

        if self.verbose:
            scores = [s for s, _ in ranked[: self.final_k]]
            print(f"[RANKER] Score medio top-{self.final_k}: {np.mean(scores):.4f}")
            print(f"[RANKER] Seleccionadas {len(results)} evidencias")

        return results


class LLMRanker:
    """
    Evalúa evidencia según probabilidad de contener una oferta económica real.
    No responde preguntas; clasifica y prioriza fragmentos.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        final_k: int = 6,
        verbose: bool = True,
    ):
        if not api_key:
            raise ValueError("Falta OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.final_k = final_k
        self.verbose = verbose

    def _build_prompt(self, evidence: List[Dict]) -> str:
        prompt = (
            "Evalúa los siguientes fragmentos.\n"
            "Tu tarea es priorizar aquellos que probablemente contienen "
            "una oferta económica presentada por un licitador.\n\n"
            "Criterios:\n"
            "- Debe tratarse de un importe ofertado, no estimado ni de referencia.\n"
            "- Puede aparecer en texto narrativo o en estructura tabular.\n"
            "- Puede haber importes con y sin impuestos.\n\n"
            "Devuelve SOLO una lista de índices ordenados por probabilidad descendente.\n\n"
        )

        for i, e in enumerate(evidence):
            meta = e["metadata"]
            snippet = e["content"][:500].replace("\n", " ")
            prompt += (
                f"[{i}] Tipo={meta.get('type')} | "
                f"Pág={meta.get('page')} | "
                f"Conf={meta.get('confidence')}\n"
                f"{snippet}\n\n"
            )

        return prompt.strip()

    def rerank(
        self,
        query: str,
        evidence: List[Dict],
    ) -> List[Dict]:
        if not evidence:
            return []

        if self.verbose:
            print(f"\n[LLM RANKER] --- Ranking contextual ({self.model}) ---")
            print(f"[LLM RANKER] Evidencias iniciales: {len(evidence)}")

        prompt = self._build_prompt(evidence)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            order = [
                int(x)
                for x in response.choices[0].message.content.split(",")
                if x.strip().isdigit()
            ]

            order = [i for i in order if i < len(evidence)]
            ranked = [evidence[i] for i in order[: self.final_k]]

            if self.verbose:
                print(f"[LLM RANKER] Orden devuelto: {order[:self.final_k]}")
                print(f"[LLM RANKER] Seleccionadas {len(ranked)} evidencias")

            return ranked

        except Exception as e:
            print(f"[LLM RANKER] Error: {e}")
            return evidence[: self.final_k]


def create_ranker(
    mode: str,
    embedder=None,
    api_key: Optional[str] = None,
    final_k: int = 10,
    verbose: bool = True,
):
    if mode == "semantic":
        if not embedder:
            raise ValueError("El ranker semántico requiere un embedder.")
        return SemanticRanker(
            embedder=embedder,
            final_k=final_k,
            verbose=verbose,
        )

    if mode == "llm":
        return LLMRanker(
            api_key=api_key,
            final_k=final_k,
            verbose=verbose,
        )

    raise ValueError("Modo de ranker no reconocido.")
