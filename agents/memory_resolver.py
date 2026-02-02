import json
from openai import OpenAI


class MemoryResolverAgent:
    """
    Decide si una query necesita acceso a la memoria conversacional previa
    para poder ser interpretada correctamente.

    NO responde la pregunta.
    NO accede a documentos.
    NO modifica el flujo del sistema.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def needs_memory(self, query: str) -> dict:
        """
        Returns:
        {
          "needs_memory": bool,
          "reason": str
        }
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=80,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a conversation dependency classifier.

                    TASK:
                    Determine whether the user's question requires access to prior conversation
                    to be correctly understood.

                    A question REQUIRES memory if:
                    - It contains references whose subject is NOT explicitly defined in the question
                    - It refers to a person, entity, value, decision or fact introduced earlier
                    - It uses pronouns, possessives or implicit references (any language)
                    - The question cannot be fully interpreted in isolation

                    A question DOES NOT require memory if:
                    - All entities it refers to are explicitly named in the question
                    - It introduces a new topic
                    - It can be correctly answered without knowing previous turns

                    IMPORTANT:
                    - Do NOT assume the reader knows who "he", "she", "it", "his", "her", "their", "su", "sus", etc. refer to
                    unless the question itself defines it.

                    Return ONLY valid JSON:
                    {
                    "needs_memory": true | false,
                    "reason": "short explanation"
                    }
                    """
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )

        raw = completion.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except Exception:
            # Fallback ultra conservador: no usar memoria
            return {
                "needs_memory": False,
                "reason": "Could not determine dependency reliably"
            }
