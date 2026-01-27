# generation/generator.py

import os
from openai import OpenAI


class Generator:
    """
    Generador de respuestas fundamentadas para un sistema RAG documental.

    PRINCIPIO CLAVE:
    - SOLO responde con información explícitamente presente en los fragmentos.
    - NO calcula, NO deriva, NO transforma valores numéricos.
    - Si la respuesta requiere cualquier tipo de cálculo o inferencia,
      debe indicar que no hay información suficiente.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Falta la variable de entorno OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        self.temperature = temperature

    def generate(self, query: str, context: str, sources: list):
        """
        Genera una respuesta EXTRACTIVA basada exclusivamente en el contexto.

        PROHIBIDO:
        - Realizar cálculos
        - Derivar métricas
        - Transformar valores numéricos
        - Inferir resultados no explícitos
        """

        print("\n[GENERATOR] ======================")
        print(f"[GENERATOR] Modelo: {self.model}")
        print(f"[GENERATOR] Pregunta: {query}")
        print(f"[GENERATOR] Nº de fuentes: {len(sources)} -> {sources}")
        print("[GENERATOR] Fragmento de contexto (vista previa 1000 caracteres):")
        print(context[:1000])
        print("[GENERATOR] ======================\n")

        # =====================================================
        # 1. Estructurar contexto
        # =====================================================
        structured_context = "\n\n".join(
            [
                f"[Fragmento {i + 1}]\n{frag.strip()}"
                for i, frag in enumerate(context.split("---"))
                if frag.strip()
            ]
        )

        max_context_chars = 18000
        if len(structured_context) > max_context_chars:
            structured_context = (
                structured_context[:max_context_chars]
                + "\n[Contexto truncado por longitud]"
            )

        # =====================================================
        # 2. PROMPT DEL SISTEMA (CONTRATO EXPLÍCITO)
        # =====================================================
        system_prompt = f"""
Eres un asistente experto en análisis documental EXTRACTIVO.

TU CONTRATO ES ESTRICTO:

- SOLO puedes responder usando información EXPLÍCITA contenida en los fragmentos.
- NO debes realizar cálculos, estadísticas, agregaciones, medias, varianzas,
  desviaciones, porcentajes ni ninguna transformación matemática.
- NO debes derivar nuevos valores a partir de datos numéricos.
- NO debes interpretar ni evaluar resultados.
- NO debes inferir información que no esté escrita literalmente.

SI la pregunta requiere:
- cálculos,
- estimaciones,
- derivaciones,
- transformaciones numéricas,
- o conclusiones no explícitas,

DEBES responder EXACTAMENTE con esta frase y nada más:

"No tengo información suficiente en los documentos proporcionados."

Fragmentos disponibles:
{structured_context}
"""

        # =====================================================
        # 3. PROMPT DEL USUARIO
        # =====================================================
        user_prompt = f"""
Pregunta del usuario:
{query}

Instrucciones finales:
- Responde de forma clara y directa.
- Usa únicamente el contenido literal de los fragmentos.
- NO expliques procesos ni razonamientos.
- Finaliza indicando las fuentes utilizadas entre paréntesis.
"""

        # =====================================================
        # 4. Llamada al modelo
        # =====================================================
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content.strip()

        # =====================================================
        # 5. Postprocesamiento y retorno
        # =====================================================
        clean_sources = sorted(set([s for s in sources if s.strip()]))

        print("\n[GENERATOR] Respuesta del modelo:\n")
        print(answer)
        print(f"[GENERATOR] Fuentes: {clean_sources}")
        print("[GENERATOR] ----------------------\n")

        return {
            "answer": answer,
            "sources": clean_sources,
        }
