from retrieval.retriever import Retriever
from retrieval.ranker import create_ranker
from generation.generator import Generator

from evaluation.judge_paper import evaluate_with_criteria
from evaluation.metrics_extended import compute_extended_metrics
from evaluation.refiner import ResponseRefiner
from evaluation.utils_logging import log_evaluation

import time
from typing import Dict, Any


class RAGQueryTool:
    """
    Tool RAG DOCUMENTAL.
    Mantiene pipeline basado en texto plano,
    pero es compatible con rankers que esperan dicts.

    El tipado semántico perezoso (lazy_typer) se usa
    SOLO como enriquecimiento del retrieval.
    """

    def __init__(
        self,
        embedder,
        vectorstore,
        generator_model: str = "gpt-4o",
        ranker_mode: str = "semantic",
        final_k: int = 6,
        lazy_typer=None,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.generator = Generator(model_name=generator_model)
        self.ranker_mode = ranker_mode
        self.final_k = final_k
        self.lazy_typer = lazy_typer

    def __call__(self, query: str, memory_context: Dict[str, Any] | None = None,):

        memory_context = memory_context or {}

        print("\n[RAG] ===============================")
        print("[RAG] Query recibida:", query)

        # ======================================================
        # 1. RETRIEVAL (con tipado perezoso opcional)
        # ======================================================
        retriever = Retriever(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
            lazy_typer=self.lazy_typer,
            enable_lazy_typing=False,
        )

        docs = retriever.retrieve(query)

        if not docs:
            return {
                "mode": "rag_documental",
                "answer": "No dispongo de documentación relevante.",
                "sources": [],
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        # ======================================================
        # 2. CONTEXTO PLANO
        # ======================================================
        context_chunks = []
        sources = []

        for d in docs:
            if not isinstance(d, dict):
                continue

            content = d.get("content")
            meta = d.get("metadata", {})

            if content:
                source = meta.get("source", "desconocido")
                page = meta.get("page")
                semantic_type = meta.get("semantic_type")

                header_parts = [f"Fuente: {source}"]
                if page is not None:
                    header_parts.append(f"página {page}")
                if semantic_type:
                    header_parts.append(f"sección {semantic_type}")

                header = " | ".join(header_parts)

                context_chunks.append(
                    f"[{header}]\n{content}"
                )

            meta = d.get("metadata", {})
            if isinstance(meta, dict):
                src = meta.get("source")
                if src:
                    sources.append(src)

        context = "\n---\n".join(context_chunks).strip()
        sources = list(set(sources))

        if not context:
            return {
                "mode": "rag_documental",
                "answer": "La documentación no contiene información suficiente.",
                "sources": [],
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        # ======================================================
        # 3. RERANKING (compatibilidad total)
        # ======================================================
        docs_raw = [d for d in context.split("\n---\n") if d.strip()]

        if not docs_raw:
            return {
                "mode": "rag_documental",
                "answer": "No hay información relevante.",
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        # WRAP: str → dict (solo para el ranker)
        docs_for_ranker = [
            {"content": d, "metadata": {}}
            for d in docs_raw
        ]

        ranker = create_ranker(
            mode=self.ranker_mode,
            embedder=self.embedder,
            final_k=self.final_k,
            verbose=False,
        )

        ranked_wrapped = ranker.rerank(query, docs_for_ranker)

        if not ranked_wrapped:
            return {
                "mode": "rag_documental",
                "answer": "No se encontró información relevante.",
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        # UNWRAP: dict → str
        ranked_docs = [
            d["content"]
            for d in ranked_wrapped
            if isinstance(d, dict) and d.get("content")
        ]

        ranked_context = "\n---\n".join(ranked_docs)

        # ======================================================
        # 4. GENERACIÓN
        # ======================================================
        start_time = time.time()

        response = self.generator.generate(
            query=query,
            context=ranked_context,
            sources=sources,
            memory_context=memory_context,
        )

        end_time = time.time()
        answer = response.get("answer", "").strip()

        if not answer:
            return {
                "mode": "rag_documental",
                "answer": "No fue posible generar una respuesta clara.",
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        # ======================================================
        # 5. EVALUACIÓN
        # ======================================================
        judge_result = evaluate_with_criteria(
            question=query,
            context=ranked_context,
            answer=answer,
        )

        metrics_ext = compute_extended_metrics(
            answer=answer,
            judge_result=judge_result,
            start_time=start_time,
            end_time=end_time,
            model_name="gpt-4o",
        )

        # ======================================================
        # 6. REFINAMIENTO
        # ======================================================
        refiner = ResponseRefiner(
            model_name="gpt-4o-mini",
            score_threshold=2.5,
        )

        refine_result = refiner.refine(
            question=query,
            answer=answer,
            judge_result=judge_result,
        )

        if refine_result.get("status") == "refined":
            answer = refine_result.get("refined_answer", answer)

        log_evaluation(
            question=query,
            answer=answer,
            judge_result=judge_result,
            metrics_ext=metrics_ext,
            refine_result=refine_result,
        )

        print("[RAG] ===============================\n")

        return {
            "mode": "rag_documental",
            "answer": answer,
            "sources": sources,
            "evaluation": judge_result,
            "refiner": refine_result,
        }

# Construcción del conjunto de herramientas RAG
def build_rag_tools(embedder, vectorstore, lazy_typer=None):
    return {
        "rag_query": RAGQueryTool(
            embedder=embedder,
            vectorstore=vectorstore,
            lazy_typer=lazy_typer,
        )
    }
