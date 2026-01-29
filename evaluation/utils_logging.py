import os
import json
import datetime
import uuid
from typing import Dict, Optional


def log_evaluation(
    question: str,
    answer: str,
    judge_result: Dict,
    metrics_ext: Dict,
    refine_result: Dict,
    output_path: str = "logs/evaluations_detailed.jsonl",
    execution_id: Optional[str] = None,
):
    """
    Guarda en disco un registro completo, auditable y trazable
    de una ejecución del sistema RAG analítico.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if execution_id is None:
        execution_id = str(uuid.uuid4())

    issue_types = set(judge_result.get("Issue_types", []))

    record = {
        "execution_id": execution_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),

        # Input / Output
        "question": question,
        "answer": answer,

        # Evaluación
        "judge": judge_result,

        # Métricas cuantitativas
        "metrics": metrics_ext,

        # Refinamiento
        "refiner": refine_result,

        # Flags derivadas (clave para análisis posterior)
        "flags": {
            "has_analytic_errors": bool(
                {"analysis_error", "wrong_calculation"} & issue_types
            ),
            "has_unsupported_inference": "unsupported_inference" in issue_types,
            "has_missing_evidence": "missing_evidence" in issue_types,
            "was_refined": refine_result.get("status") == "refined",
            "is_safe_answer": (
                judge_result.get("Faithfulness", 0) >= 3
                and not {"analysis_error", "unsupported_inference"} & issue_types
            ),
        },
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
