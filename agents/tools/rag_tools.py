# agentic/tools/rag_tools.py

from retrieval.retriever import Retriever
from retrieval.ranker import create_ranker
from generation.generator import Generator

# Evaluación y refinamiento
from evaluation.judge_paper import evaluate_with_criteria
from evaluation.metrics_extended import compute_extended_metrics
from evaluation.refiner import ResponseRefiner
from evaluation.utils_logging import log_evaluation

import time


class RAGQueryTool:
    """
    Tool RAG DOCUMENTAL.

    Ejecuta el pipeline completo:
      1. Retrieval
      2. Reranking
      3. Generación con grounding
      4. Evaluación
      5. Refinamiento editorial (opcional)
      6. Logging

    NO calcula métricas numéricas.
    NO infiere datos no presentes en el contexto.
    """

    def __init__(
        self,
        embedder,
        vectorstore,
        generator_model: str = "gpt-4o",
        ranker_mode: str = "semantic",
        final_k: int = 6,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.generator = Generator(model_name=generator_model)
        self.ranker_mode = ranker_mode
        self.final_k = final_k

    def __call__(self, query: str):
        """
        Ejecuta el pipeline RAG documental.

        Devuelve un payload enriquecido, apto para auditoría.
        """

        print("\n[RAG] ===============================")
        print("[RAG] Query recibida:", query)

        # ======================================================
        # 1. RETRIEVAL
        # ======================================================
        retriever = Retriever(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )

        docs = retriever.retrieve(query)

        if not docs:
            return {
                "mode": "rag_documental",
                "answer": (
                    "No dispongo de documentación cargada o relevante para "
                    "responder a esta pregunta."
                ),
                "sources": [],
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        # ------------------------------------------------------
        # Construcción explícita de contexto y fuentes
        # ------------------------------------------------------
        context_chunks = []
        sources = []

        for d in docs:
            if not isinstance(d, dict):
                continue

            content = d.get("content")
            if content:
                context_chunks.append(content)

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
                "answer": (
                    "La documentación disponible no contiene información "
                    "suficiente para responder a esta consulta."
                ),
                "sources": [],
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        # ======================================================
        # 2. RERANKING
        # ======================================================
        docs_raw = [d for d in context.split("\n---\n") if d.strip()]
        if not docs_raw:
            return {
                "mode": "rag_documental",
                "answer": (
                    "La documentación disponible no contiene información "
                    "suficiente para responder a esta consulta."
                ),
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        ranker = create_ranker(
            mode=self.ranker_mode,
            embedder=self.embedder,
            final_k=self.final_k,
            verbose=False,
        )

        ranked_docs = [d for d in ranker.rerank(query, docs_raw) if d.strip()]
        if not ranked_docs:
            return {
                "mode": "rag_documental",
                "answer": (
                    "No se encontró información relevante en la documentación "
                    "para responder a esta pregunta."
                ),
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
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

        if not answer:
            return {
                "mode": "rag_documental",
                "answer": (
                    "No fue posible generar una respuesta clara a partir "
                    "de la documentación disponible."
                ),
                "sources": sources,
                "evaluation": None,
                "metrics": None,
                "refiner": None,
            }

        # ======================================================
        # 4. EVALUACIÓN
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
        # 5. REFINAMIENTO EDITORIAL
        # ======================================================
        refiner = ResponseRefiner(
            model_name="gpt-4o-mini",
            threshold=2.5,
        )

        refine_result = refiner.refine(
            question=query,
            context=ranked_context,
            answer=answer,
            judge_result=judge_result,
        )

        if refine_result.get("status") == "refined":
            answer = refine_result.get("refined_answer", answer)

        # ======================================================
        # 6. LOGGING
        # ======================================================
        log_evaluation(
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
            "metrics": metrics_ext,
            "refiner": refine_result,
        }


def build_rag_tools(embedder, vectorstore):
    """
    Construye las herramientas RAG expuestas al ToolManager.
    """

    rag_query_tool = RAGQueryTool(embedder, vectorstore)

    return {
        "rag_query": rag_query_tool
    }
