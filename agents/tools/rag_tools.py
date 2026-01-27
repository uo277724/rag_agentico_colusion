# agentic/tools/rag_tools.py

from retrieval.retriever import Retriever
from retrieval.ranker import create_ranker
from generation.generator import Generator

# === Evaluación y refinamiento ===
from evaluation.judge_paper import evaluate_with_criteria
from evaluation.metrics_extended import compute_extended_metrics
from evaluation.refiner import ResponseRefiner
from evaluation.utils_logging import log_evaluation

import time


class RAGQueryTool:
    """
    Tool única que ejecuta el pipeline completo de RAG:
      1. Recuperación
      2. Re-ranking
      3. Generación final con grounding
      4. Evaluación + refinamiento (opcional)
    """

    def __init__(self, embedder, vectorstore, generator_model="gpt-4o"):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.generator = Generator(model_name=generator_model)

    def __call__(self, query: str):
        """
        Ejecuta el pipeline completo del RAG.

        SIEMPRE devuelve:
        {
            "answer": str,
            "sources": list
        }
        """

        print("\n[RAG] ===============================")
        print("[RAG] Query recibida:", query)

        # ======================================================
        # 1. RETRIEVAL
        # ======================================================
        retriever = Retriever(self.embedder, self.vectorstore, top_k=25)
        results = retriever.retrieve(query)

        context = results.get("context", "").strip()
        sources = results.get("sources", [])

        # ------------------------------------------------------
        # GUARD 1: No hay documentación
        # ------------------------------------------------------
        if not context:
            print("[RAG] No se recuperó ningún fragmento relevante")

            return {
                "answer": (
                    "No dispongo de documentación cargada o relevante para "
                    "responder a esta pregunta en este momento."
                ),
                "sources": []
            }

        # ======================================================
        # 2. PREPARACIÓN + RERANKING
        # ======================================================
        docs_raw = [d for d in context.split("\n---\n") if d.strip()]

        # ------------------------------------------------------
        # GUARD 2: Contexto inválido tras el split
        # ------------------------------------------------------
        if not docs_raw:
            print("[RAG] El contexto se ha quedado vacío tras el preprocesado")

            return {
                "answer": (
                    "La documentación disponible no contiene información "
                    "suficiente para responder a esta consulta."
                ),
                "sources": []
            }

        ranker = create_ranker(
            mode="semantic",
            embedder=self.embedder,
            final_k=6,
            verbose=False,
        )

        ranked_docs = ranker.rerank(query, docs_raw)
        ranked_docs = [d for d in ranked_docs if d.strip()]

        # ------------------------------------------------------
        # GUARD 3: Reranking sin resultados útiles
        # ------------------------------------------------------
        if not ranked_docs:
            print("[RAG] El reranker no devolvió documentos útiles")

            return {
                "answer": (
                    "No se encontró información relevante en la documentación "
                    "para responder a esta pregunta."
                ),
                "sources": []
            }

        ranked_context = "\n---\n".join(ranked_docs)

        # ======================================================
        # 3. GENERACIÓN
        # ======================================================
        start_time = time.time()

        response = self.generator.generate(
            query=query,
            context=ranked_context,
            sources=sources,
        )

        end_time = time.time()

        answer = response.get("answer", "").strip()

        # ------------------------------------------------------
        # GUARD 4: El generador devolvió vacío
        # ------------------------------------------------------
        if not answer:
            print("[RAG] El generador devolvió una respuesta vacía")

            return {
                "answer": (
                    "No fue posible generar una respuesta clara a partir "
                    "de la documentación disponible."
                ),
                "sources": sources
            }

        print("[RAG] Respuesta antes del refiner:\n", answer)

        # ======================================================
        # 4. EVALUACIÓN + MÉTRICAS
        # ======================================================
        judge_result = evaluate_with_criteria(
            question=query,
            context=ranked_context,
            answer=answer
        )

        metrics_ext = compute_extended_metrics(
            answer=answer,
            judge_result=judge_result,
            start_time=start_time,
            end_time=end_time,
            model_name="gpt-4o"
        )

        # ======================================================
        # 5. REFINER (opcional)
        # ======================================================
        refiner = ResponseRefiner(
            model_name="gpt-4o-mini",
            threshold=2.5
        )

        refine_result = refiner.refine(
            question=query,
            context=ranked_context,
            answer=answer,
            judge_result=judge_result
        )

        if refine_result.get("status") == "refined":
            answer = refine_result.get("refined_answer", answer)
            print("[RAG] Respuesta refinada aplicada")

        # ======================================================
        # 6. LOGGING
        # ======================================================
        log_evaluation(
            judge_result=judge_result,
            metrics_ext=metrics_ext,
            refine_result=refine_result
        )

        print("[RAG] Estado del refiner:", refine_result.get("status"))
        print("[RAG] Respuesta final:\n", answer)
        print("[RAG] ===============================\n")

        return {
            "answer": answer,
            "sources": sources
        }


def build_rag_tools(embedder, vectorstore):
    """
    Construye las herramientas RAG expuestas al planner.
    """
    rag_query_tool = RAGQueryTool(embedder, vectorstore)

    return {
        "rag_query": rag_query_tool
    }
