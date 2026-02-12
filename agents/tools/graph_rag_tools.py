from retrieval.ranker import create_ranker
from generation.generator import Generator
from graph.retriever import GraphRetriever

from evaluation.judge_paper import evaluate_with_criteria
from evaluation.metrics_extended import compute_extended_metrics
from evaluation.refiner import ResponseRefiner
from evaluation.utils_logging import log_evaluation

import time
from typing import Dict, Any


class GraphRAGQueryTool:
    """
    GraphRAG tool that expands vector hits with graph context.
    """

    def __init__(
        self,
        embedder,
        vectorstore,
        graph_store,
        generator_model: str = "gpt-4o",
        ranker_mode: str = "semantic",
        final_k: int = 6,
        lazy_typer=None,
        max_total: int = 20,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.graph_store = graph_store
        self.generator = Generator(model_name=generator_model)
        self.ranker_mode = ranker_mode
        self.final_k = final_k
        self.lazy_typer = lazy_typer
        self.max_total = max_total

    def __call__(self, query: str, memory_context: Dict[str, Any] | None = None):
        memory_context = memory_context or {}

        print("\n[GRAPH_RAG] ===============================")
        print("[GRAPH_RAG] Query recibida:", query)

        retriever = GraphRetriever(
            graph_store=self.graph_store,
            embedder=self.embedder,
            vectorstore=self.vectorstore,
            top_k_primary=self.final_k,
            max_total=self.max_total,
            lazy_typer=self.lazy_typer,
            enable_lazy_typing=False,
        )

        docs = retriever.retrieve(query)
        if not docs:
            return {
                "mode": "graph_rag",
                "answer": "No dispongo de documentacion relevante.",
                "sources": [],
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

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
                    header_parts.append(f"pagina {page}")
                if semantic_type:
                    header_parts.append(f"seccion {semantic_type}")

                header = " | ".join(header_parts)

                context_chunks.append(
                    f"[{header}]\n{content}"
                )

            if isinstance(meta, dict):
                src = meta.get("source")
                if src:
                    sources.append(src)

        context = "\n---\n".join(context_chunks).strip()
        sources = list(set(sources))

        if not context:
            return {
                "mode": "graph_rag",
                "answer": "La documentacion no contiene informacion suficiente.",
                "sources": [],
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        docs_raw = [d for d in context.split("\n---\n") if d.strip()]
        if not docs_raw:
            return {
                "mode": "graph_rag",
                "answer": "No hay informacion relevante.",
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

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
                "mode": "graph_rag",
                "answer": "No se encontro informacion relevante.",
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        ranked_docs = [
            d["content"]
            for d in ranked_wrapped
            if isinstance(d, dict) and d.get("content")
        ]

        ranked_context = "\n---\n".join(ranked_docs)

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
                "mode": "graph_rag",
                "answer": "No fue posible generar una respuesta clara.",
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

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

        print("[GRAPH_RAG] ===============================\n")

        return {
            "mode": "graph_rag",
            "answer": answer,
            "sources": sources,
            "evaluation": judge_result,
            "refiner": refine_result,
        }


def build_graph_rag_tools(embedder, vectorstore, graph_store, lazy_typer=None):
    return {
        "graph_rag_query": GraphRAGQueryTool(
            embedder=embedder,
            vectorstore=vectorstore,
            graph_store=graph_store,
            lazy_typer=lazy_typer,
        )
    }
