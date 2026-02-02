# generation/generator.py

import os
from typing import Optional, List
from openai import OpenAI


class Generator:
    """
    Generador de respuestas fundamentadas para un sistema RAG multimodal y generalista.
    Usa modelos de lenguaje (por defecto GPT-4o) para producir respuestas precisas y contextualizadas.
    Garantiza grounding: solo se apoya en la información de los fragmentos proporcionados.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Falta la variable de entorno OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        self.temperature = temperature

    def generate(
        self,
        query: str,
        context: str,
        sources: List[str],
        memory_context: Optional[str] = None,
    ):
        """
        Genera una respuesta fundamentada en el contexto proporcionado.
        La memoria conversacional se usa SOLO para coherencia, nunca como fuente factual.
        """

        print("\n[GENERATOR] ======================")
        print(f"[GENERATOR] Modelo: {self.model}")
        print(f"[GENERATOR] Pregunta: {query}")
        print(f"[GENERATOR] Nº de fuentes: {len(sources)} -> {sources}")

        if memory_context:
            print("[GENERATOR] Memory context recibido:")
            print(memory_context)

        print("[GENERATOR] Fragmento de contexto (vista previa 1000 caracteres):")
        print(context[:1000])
        print("[GENERATOR] ======================\n")

        # =====================================================
        # 1. Estructurar contexto documental (IGUAL QUE ANTES)
        # =====================================================
        structured_context = "\n\n".join(
            [
                f"[Fragmento {i+1} | Fuente: {sources[i] if i < len(sources) else 'desconocida'}]\n"
                f"{frag.strip()}"
                for i, frag in enumerate(context.split('---'))
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
        # 2. Bloque de memoria (NUEVO, AISLADO)
        # =====================================================
        memory_block = ""
        if memory_context:
            memory_block = (
                "\n\nContexto de la conversación previa "
                "(solo para coherencia y repreguntas, NO como fuente factual):\n"
                f"{memory_context}\n"
            )

        # =====================================================
        # 3. Prompt del sistema (MISMO + REGLAS DE MEMORIA)
        # =====================================================
        system_prompt = (
            "Eres un asistente experto en análisis y comprensión documental. "
            "RESPONDES SIEMPRE con trazabilidad documental. "
            "Cuando proporciones una respuesta, DEBES indicar explícitamente: "
            "- el documento de origen, "
            "- el número de página (si aparece en el fragmento), "
            "- y el tipo de sección o tabla si se menciona.\n\n"

            "REGLAS CRÍTICAS:\n"
            "- Usa EXCLUSIVAMENTE los fragmentos documentales como fuente factual.\n"
            "- La memoria conversacional es SOLO para coherencia lingüística y referencia implícita.\n"
            "- NO derives hechos ni datos nuevos a partir de la memoria.\n"
            "- NO inventes referencias, páginas ni cargos.\n\n"

            "CUANDO LA INFORMACIÓN NO SEA SUFICIENTE:\n"
            "- NO inventes una respuesta.\n"
            "- Explica claramente QUÉ información concreta no aparece en los documentos.\n"
            "- Indica brevemente QUÉ información relacionada SÍ está presente, si la hay.\n"
            "- Sugiere UNA forma concreta de reformular o concretar la pregunta para poder responder.\n"
            "- No finalices la respuesta con una negativa genérica.\n\n"

            f"{memory_block}\n"
            "Fragmentos relevantes:\n"
            f"{structured_context}"
        )

        # =====================================================
        # 4. Prompt del usuario (IGUAL)
        # =====================================================
        user_prompt = (
            f"Pregunta del usuario: {query}\n\n"
            "Redacta una respuesta precisa, clara y basada exclusivamente en los fragmentos anteriores. "
            "Si la respuesta se apoya en contenido visual descrito (como diagramas o esquemas), "
            "menciónalo brevemente sin afirmar cosas no observadas. "
            "Indica explícitamente en el texto de la respuesta "
            "de qué documento, página o fragmento procede cada afirmación relevante. "
            "No uses referencias genéricas como 'Fragmento X' si hay información más concreta."
        )

        # =====================================================
        # 5. Llamada al modelo
        # =====================================================
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content.strip()

        # =====================================================
        # 6. Postprocesamiento (IGUAL)
        # =====================================================
        clean_sources = sorted(set([s for s in sources if s.strip()]))
        answer_final = answer.strip()

        print("\n[GENERATOR] Respuesta del modelo:\n")
        print(answer_final)
        print(f"[GENERATOR] Fuentes: {clean_sources}")
        print("[GENERATOR] ----------------------\n")

        return {
            "answer": answer_final,
            "sources": clean_sources,
        }
