# agentic/tool_manager.py

import json

class ToolManager:
    """
    Registra herramientas (funciones) y permite ejecutarlas desde el Planner.
    """

    def __init__(self):
        self.tools = {}   # nombre → función Python

    # --------------------------------------------------------
    # Registro de herramientas
    # --------------------------------------------------------
    def register(self, name: str, func):
        """
        name: nombre con el que el LLM llamará a la herramienta
        func: función Python que implementa la herramienta
        """
        if not callable(func):
            raise ValueError(f"La herramienta '{name}' no es una función.")
        self.tools[name] = func

    # --------------------------------------------------------
    # Ejecución
    # --------------------------------------------------------
    def execute(self, name: str, args: dict):
        print(f"DEBUG: ToolManager executing {name} with args: {args}")
        if name not in self.tools:
            raise ValueError(f"Herramienta desconocida: {name}")

        try:
            result = self.tools[name](**args)
            print(f"DEBUG: Tool {name} returned:", result)
            return {"ok": True, "result": result}
        except Exception as e:
            print(f"DEBUG: Tool {name} raised exception:", e)
            return {"ok": False, "error": str(e)}

