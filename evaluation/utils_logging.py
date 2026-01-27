# evaluation/utils_logging.py

import os
import json
import datetime

def log_evaluation(judge_result, metrics_ext, refine_result, output_path="logs/evaluations_detailed.jsonl"):
    """
    Guarda en disco los resultados completos de evaluación, métricas y refinamiento.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "judge": judge_result,
        "metrics": metrics_ext,
        "refiner": refine_result
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
