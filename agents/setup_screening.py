# agentic/setup_screening.py

from agents.tool_manager import ToolManager

# RAG tools
from agents.tools.rag_tools import build_rag_tools
from agents.tools.rag_extract_bids import RAGExtractBidsTool

# Calculation Agents
from agents.calculation.cv import CVAgent
from agents.calculation.spd import SPDAgent
from agents.calculation.diffp import DIFFPAgent
from agents.calculation.rd import RDAgent
from agents.calculation.kurt import KurtosisAgent
from agents.calculation.skew import SkewnessAgent
from agents.calculation.kstest import KSTestAgent

# Planner
from agents.planner import ScreeningPlannerAgent

# Lazy Semantic Typer
from retrieval.lazy_semantic_typer import LazySemanticTyper

# Inicialización del Lazy Semantic Typer
lazy_typer = LazySemanticTyper(
    model="gpt-4o-mini",
    max_chunks=20,
    debug=True,
)

def initialize_screening_agentic(embedder, vectorstore):
    """
    Inicializa el sistema agéntico de screening de licitaciones.

    Responsabilidades del sistema:
    - RAG documental (explicativo)
    - RAG extractivo (bids)
    - Cálculo determinista de métricas
    - Interpretación no acusatoria de resultados
    """

    # ---------------------------
    # SYSTEM PROFILE (DECLARATIVO)
    # ---------------------------
    system_profile = {
        "system_type": "public_tender_screening",
        "mode": "screening",
        "supported_metrics": [
            "cv", "spd", "diffp", "rd", "kurt", "skew", "kstest"
        ],
        "rag_capabilities": [
            "documentation_query",
            "bid_extraction"
        ],
    }

    # ---------------------------
    # TOOL MANAGER
    # ---------------------------
    tool_manager = ToolManager(debug=True)

    # RAG narrativo
    rag_tools = build_rag_tools(embedder, vectorstore, lazy_typer=lazy_typer)
    tool_manager.register("rag_query", rag_tools["rag_query"])

    # RAG extractivo (bids)
    rag_extract_bids_tool = RAGExtractBidsTool(embedder, vectorstore , lazy_typer=lazy_typer)
    tool_manager.register("rag_extract_bids", rag_extract_bids_tool)

    # ---------------------------
    # CALCULATION AGENTS (DETERMINISTAS)
    # ---------------------------
    calculation_agents = {
        "cv": CVAgent(),
        "spd": SPDAgent(),
        "diffp": DIFFPAgent(),
        "rd": RDAgent(),
        "kurt": KurtosisAgent(),
        "skew": SkewnessAgent(),
        "kstest": KSTestAgent(),
    }

    # ---------------------------
    # PLANNER (ORQUESTADOR)
    # ---------------------------
    planner = ScreeningPlannerAgent(
        tool_manager=tool_manager,
        calculation_agents=calculation_agents
    )

    # ---------------------------
    # EXPORTAR SISTEMA
    # ---------------------------
    return {
        "planner": planner,
        "tool_manager": tool_manager,
        "calculation_agents": calculation_agents,
        "rag_tools": rag_tools,
        "system_profile": system_profile,
    }
