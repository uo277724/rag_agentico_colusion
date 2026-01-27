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
from embeddings import embedder
import vectorstore


def initialize_screening_agentic(embedder, vectorstore):
    """
    Inicializa el sistema Agentic para Screening de licitaciones:
    - RAG documental narrativo
    - RAG numérico (extracción de bids)
    - Cálculo determinista de métricas
    """

    # ---------------------------
    # TOOL MANAGER
    # ---------------------------
    tool_manager = ToolManager()

    # RAG narrativo
    rag_tools = build_rag_tools(embedder, vectorstore)
    tool_manager.register("rag_query", rag_tools["rag_query"])

    # RAG numérico
    rag_extract_bids_tool = RAGExtractBidsTool(embedder, vectorstore)
    tool_manager.register("rag_extract_bids", rag_extract_bids_tool)

    # ---------------------------
    # CALCULATION AGENTS
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
    # PLANNER
    # ---------------------------
    planner = ScreeningPlannerAgent(
        tool_manager=tool_manager,
        calculation_agents=calculation_agents
    )

    # ---------------------------
    # DEVOLVER COMPONENTES
    # ---------------------------
    return {
        "planner": planner,
        "tool_manager": tool_manager,
        "calculation_agents": calculation_agents,
        "rag_tools": rag_tools,
    }
