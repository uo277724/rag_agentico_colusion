# retrieval/ranker.py

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class SemanticRanker:
    """
    Reordena documentos según similitud de embeddings con la consulta.
    Neutral respecto al tipo de contenido (texto, figura, descripción, etc.).
    Ideal tras una recuperación inicial amplia (top_k alto).
    """

    def __init__(self, embedder, final_k: int = 15, verbose: bool = True):
        self.embedder = embedder
        self.final_k = final_k
        self.verbose = verbose

    def _clean_docs(self, docs: list) -> list:
        """
        Limpieza ligera: elimina ruido visual y normaliza espacios,
        pero sin borrar información semántica (como descripciones o metadatos).
        """
        cleaned = []
        for d in docs:
            if not isinstance(d, str):
                continue
            text = re.sub(r"\s{2,}", " ", d)
            text = text.replace("\r", " ").replace("\n", " ").strip()
            if len(text) > 10:
                cleaned.append(text)
        return cleaned

    def rerank(self, query: str, docs: list):
        """
        Reordena los documentos por similitud de embeddings.
        Devuelve los top N (final_k) fragmentos más relevantes.
        """
        if not docs:
            print("[RANKER] ⚠️ No hay documentos para reordenar.")
            return []

        if self.verbose:
            print(f"\n[RANKER] --- Reordenamiento semántico ---")
            print(f"[RANKER] Fragmentos iniciales: {len(docs)}")

        docs = self._clean_docs(docs)

        # Embeddings
        query_emb = self.embedder.embed_texts([query])[0]
        doc_embs = self.embedder.embed_texts(docs)
        sims = cosine_similarity([query_emb], doc_embs)[0]

        # Orden descendente
        ranked = sorted(zip(docs, sims), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:min(self.final_k, len(ranked))]]

        if self.verbose:
            top_sim_mean = np.mean(sorted(sims, reverse=True)[:min(self.final_k, len(sims))])
            print(f"[RANKER] Similitud media top {self.final_k}: {top_sim_mean:.4f}")
            print(f"[RANKER] Seleccionados {len(top_docs)} fragmentos más relevantes.")
            print("[RANKER] -----------------------------\n")

        return top_docs


class LLMRanker:
    """
    Reordena documentos según evaluación contextual del modelo de lenguaje.
    Permite capturar relevancia semántica completa (ideal para preguntas abiertas o ambiguas).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", final_k: int = 6, verbose: bool = True):
        if not api_key:
            raise ValueError("Falta la variable de entorno OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.final_k = final_k
        self.verbose = verbose

    def _format_prompt(self, query: str, docs: list) -> str:
        """
        Construye un prompt claro y neutro para ranking contextual.
        """
        prompt = (
            "Eres un asistente experto en comprensión de información multimodal.\n"
            f"Debes ordenar los siguientes fragmentos del más al menos relevante "
            f"para responder la pregunta: '{query}'.\n"
            "Devuelve solo una lista de índices en orden de relevancia (por ejemplo: 2,0,1,...).\n\n"
        )

        for i, d in enumerate(docs):
            snippet = re.sub(r"\s{2,}", " ", d.replace("\n", " ")).strip()
            prompt += f"[{i}] {snippet[:500]}...\n\n"
        return prompt.strip()

    def rerank(self, query: str, docs: list):
        if not docs:
            print("[LLM RANKER] ⚠️ No hay documentos para reordenar.")
            return []

        if self.verbose:
            print(f"\n[LLM RANKER] --- Reordenando con {self.model} ---")
            print(f"[LLM RANKER] Fragmentos iniciales: {len(docs)}")

        prompt = self._format_prompt(query, docs)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            order = [int(x) for x in re.findall(r"\d+", response.choices[0].message.content)]
            order = [i for i in order if i < len(docs)]

            ranked_docs = [docs[i] for i in order[:self.final_k]]

            if self.verbose:
                print(f"[LLM RANKER] Orden sugerido: {order[:self.final_k]}")
                print(f"[LLM RANKER] Seleccionados {len(ranked_docs)} fragmentos más relevantes.")
                print("[LLM RANKER] -----------------------------\n")

            return ranked_docs

        except Exception as e:
            print(f"[LLM RANKER] ⚠️ Error al reordenar con LLM: {e}")
            print("[LLM RANKER] Se usará el orden original.\n")
            return docs[:self.final_k]


def create_ranker(mode: str, embedder=None, api_key: str = None, final_k: int = 10, verbose: bool = True):
    """
    Crea el ranker adecuado según el modo seleccionado:
      - 'semantic' → ranking por similitud de embeddings
      - 'llm'      → ranking contextual con modelo de lenguaje
    """
    if mode == "semantic":
        if not embedder:
            raise ValueError("El ranker semántico requiere un embedder.")
        return SemanticRanker(embedder=embedder, final_k=final_k, verbose=verbose)
    elif mode == "llm":
        return LLMRanker(api_key=api_key, final_k=final_k, verbose=verbose)
    else:
        raise ValueError("Modo de ranker no reconocido: usa 'semantic' o 'llm'.")