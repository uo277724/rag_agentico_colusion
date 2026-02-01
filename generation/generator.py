# generation/generator.py

import os
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

    def generate(self, query: str, context: str, sources: list):
        """
        Genera una respuesta fundamentada en el contexto proporcionado.
        No inventa información fuera de los fragmentos.
        Puede trabajar con descripciones textuales y visuales.
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
                f"[Fragmento {i+1} | Fuente: {sources[i] if i < len(sources) else 'desconocida'}]\n"
                f"{frag.strip()}"
                for i, frag in enumerate(context.split('---'))
                if frag.strip()
            ]
        )

        max_context_chars = 18000
        if len(structured_context) > max_context_chars:
            structured_context = structured_context[:max_context_chars] + "\n[Contexto truncado por longitud]"

        # =====================================================
        # 2. Prompt del sistema y del usuario
        # =====================================================
        system_prompt = (
            "Eres un asistente experto en análisis y comprensión documental. "
            "RESPONDES SIEMPRE con trazabilidad documental. "
            "Cuando proporciones una respuesta, DEBES indicar explícitamente: "
            "- el documento de origen, "
            "- el número de página (si aparece en el fragmento), "
            "- y el tipo de sección o tabla si se menciona. "
            "No inventes referencias ni páginas. "
            "Tu tarea es responder preguntas basándote únicamente en los fragmentos proporcionados. "
            "Los fragmentos pueden contener texto descriptivo, información técnica o descripciones de elementos visuales. "
            "Analiza su contenido globalmente, identifica las partes relevantes y genera una respuesta clara y fundamentada. "
            "No inventes información ni hagas suposiciones fuera de lo que está explícitamente en el contexto. "
            "Si no hay suficiente información, responde exactamente: "
            "'No tengo información suficiente en los documentos proporcionados.'\n\n"
            "Fragmentos relevantes:\n"
            f"{structured_context}"
        )

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
        # 3. Llamada al modelo
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
        # 4. Postprocesamiento y retorno (sin repetir fuentes)
        # =====================================================
        clean_sources = sorted(set([s for s in sources if s.strip()]))
        answer_final = answer.strip()

        print("\n[GENERATOR] Respuesta del modelo:\n")
        print(answer_final)
        print(f"[GENERATOR] Fuentes: {clean_sources}")
        print("[GENERATOR] ----------------------\n")

        return {"answer": answer_final, "sources": clean_sources}