import json
import re
import os
from datetime import datetime
from openai import OpenAI


class RAGJudge:
    """
    Agente evaluador (LLM-as-a-Judge) para medir la calidad de las respuestas del RAG.
    Registra métricas en un archivo externo (CSV o JSONL) para análisis posterior.
    """

    def __init__(self,
                 model_name="gpt-4o-mini",
                 temperature=0.0,
                 log_dir="logs",
                 log_format="jsonl"):
        print(f"[JUDGE] Inicializando juez con modelo {model_name}")
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, f"evaluations.{log_format}")
        self.log_format = log_format.lower()

        print(f"[JUDGE] Resultados se guardarán en: {self.log_file}")

    # ---------------------------------------------------------------------
    def _build_prompt(self, question: str, context: str, answer: str) -> str:
        return f"""
        Eres un evaluador experto en documentación técnica.
        Evalúa la calidad de la siguiente respuesta según el contexto.

        Devuelve solo un JSON con tres métricas entre 0.0 y 1.0:
        - factual_accuracy: veracidad y coherencia con el contexto.
        - informativeness: cobertura informativa relevante.
        - clarity: claridad y concisión de la redacción.

        Ejemplo:
        {{
          "factual_accuracy": 0.94,
          "informativeness": 0.88,
          "clarity": 0.80
        }}

        Pregunta: {question}

        Contexto (recortado):
        {context[:4000]}

        Respuesta:
        {answer}
        """.strip()

    # ---------------------------------------------------------------------
    def evaluate(self, question: str, context: str, answer: str):
        """Evalúa una respuesta y guarda los resultados en archivo."""
        prompt = self._build_prompt(question, context, answer)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=300,
            )

            raw_text = completion.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not match:
                raise ValueError("No se detectó JSON válido en la respuesta.")

            metrics = json.loads(match.group(0))
            metrics = {k: float(metrics.get(k, 0.0)) for k in
                       ["factual_accuracy", "informativeness", "clarity"]}

            record = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "factual_accuracy": metrics["factual_accuracy"],
                "informativeness": metrics["informativeness"],
                "clarity": metrics["clarity"]
            }

            self._log_metrics(record)
            print(f"[JUDGE] Métricas registradas: {metrics}")
            return metrics

        except Exception as e:
            print(f"[JUDGE] Error evaluando respuesta: {e}")
            return {"factual_accuracy": 0.0, "informativeness": 0.0, "clarity": 0.0}

    # ---------------------------------------------------------------------
    def _log_metrics(self, record):
        """Guarda cada evaluación en un archivo persistente."""
        if self.log_format == "jsonl":
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        elif self.log_format == "csv":
            import csv
            file_exists = os.path.isfile(self.log_file)
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(record.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
