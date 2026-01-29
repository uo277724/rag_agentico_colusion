# agentic/tool_manager.py

from typing import Callable, Dict, Any, Optional


class ToolManager:
    """
    Registro y ejecución controlada de herramientas deterministas.
    Actúa como frontera semántica entre agentes LLM y código Python.
    """

    def __init__(self, debug: bool = False):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.debug = debug

    # --------------------------------------------------------
    # Registro de herramientas
    # --------------------------------------------------------
    def register(
        self,
        name: str,
        func: Callable,
        tool_type: str = "generic",
        required_keys: Optional[list] = None,
        description: str = "",
    ):
        """
        Registra una herramienta con metadatos explícitos.

        - name: identificador interno
        - func: callable Python
        - tool_type: rag | extract | calculate | generic
        - required_keys: claves mínimas esperadas en args
        """

        if not callable(func):
            raise ValueError(f"La herramienta '{name}' no es callable.")

        self.tools[name] = {
            "func": func,
            "type": tool_type,
            "required_keys": required_keys or [],
            "description": description,
        }

        if self.debug:
            print(f"[TOOL MANAGER] Registrada herramienta '{name}' ({tool_type})")

    # --------------------------------------------------------
    # Ejecución
    # --------------------------------------------------------
    def execute(self, name: str, args: dict) -> Dict[str, Any]:
        if self.debug:
            print(f"\n[TOOL MANAGER] =====================")
            print(f"[TOOL MANAGER] Ejecutando herramienta: {name}")
            print(f"[TOOL MANAGER] Args recibidos: {args}")

        if name not in self.tools:
            if self.debug:
                print(f"[TOOL MANAGER] ❌ Herramienta desconocida: {name}")
            return {
                "ok": False,
                "error_type": "unknown_tool",
                "message": f"Herramienta desconocida: {name}",
            }

        tool = self.tools[name]

        # -----------------------------
        # Validación mínima de argumentos
        # -----------------------------
        missing = [
            k for k in tool["required_keys"]
            if k not in args
        ]
        if missing:
            if self.debug:
                print(f"[TOOL MANAGER] ❌ Argumentos faltantes: {missing}")
            return {
                "ok": False,
                "error_type": "invalid_arguments",
                "message": f"Faltan argumentos requeridos: {missing}",
                "tool": name,
            }

        # -----------------------------
        # Ejecución protegida
        # -----------------------------
        try:
            result = tool["func"](**args)

            if self.debug:
                print(f"[TOOL MANAGER] ✔ Tool '{name}' ejecutada correctamente")
                print(f"[TOOL MANAGER] Resultado tipo: {type(result)}")

            return {
                "ok": True,
                "result": result,
                "tool": name,
                "tool_type": tool["type"],
            }

        except Exception as e:
            # ERROR DURO → propagación clara
            if self.debug:
                print(f"[TOOL MANAGER] ❌ Excepción en tool '{name}': {repr(e)}")

            return {
                "ok": False,
                "error_type": "execution_error",
                "message": str(e),
                "tool": name,
            }

    # --------------------------------------------------------
    # Introspección (opcional)
    # --------------------------------------------------------
    def list_tools(self):
        """
        Devuelve metadatos de herramientas registradas.
        Útil para auditoría y debugging.
        """
        return {
            name: {
                "type": t["type"],
                "required_keys": t["required_keys"],
                "description": t["description"],
            }
            for name, t in self.tools.items()
        }
