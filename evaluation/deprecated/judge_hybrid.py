# evaluation/judge_hybrid.py

import os
import json
import datetime
from evaluation.deprecated.metrics import compute_numeric_metrics, compute_llm_metrics, aggregate_metrics


class HybridJudge:
    """
    Evaluador híbrido de rendimiento RAG: combina métricas cuantitativas y semánticas.
    """

    def __init__(self, embedder, model_name="gpt-4o-mini", log_path="logs/evaluations.jsonl"):
        self.embedder = embedder
        self.model_name = model_name
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def evaluate(self, question, context, answer):
        """
        Ejecuta ambas evaluaciones (numérica + LLM), las combina y guarda los resultados.
        """
        numeric = compute_numeric_metrics(question, context, answer, self.embedder)
        llm = compute_llm_metrics(question, context, answer, self.model_name)
        hybrid = aggregate_metrics(numeric, llm)

        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "question": question,
            "numeric": numeric,
            "llm": llm,
            "hybrid": hybrid
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record
