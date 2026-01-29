import os
import time
import streamlit as st

# ==========================================================
# INGESTIÓN + EMBEDDINGS
# ==========================================================
from ingestion.loader import process_file
from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaVectorStore

# ==========================================================
# AGENTIC SETUP (SCREENING ONLY)
# ==========================================================
from agents.setup_screening import initialize_screening_agentic


# ==========================================================
# UI HELPERS
# ==========================================================
def chat_container():
    return st.container(border=True)

def render_user_message(text: str):
    col_left, col_right = st.columns([1, 4])
    with col_right:
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="
                    text-align: right;
                    margin-right: 20px;
                    padding-bottom: 12px;
                    padding-top: 2px;
                ">
                    <div style="font-weight: 600;">🧑 Usuario</div>
                    <div>{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def render_screening_result(result: dict):
    metrics = result.get("metrics", {})
    explanation = result.get("explanation")
    meta = result.get("meta", {})

    col_left, col_right = st.columns([4, 1])
    with col_left:
        with st.container(border=True):
            st.markdown("🤖 **Asistente**")

            if not metrics:
                st.warning("No se han devuelto métricas.")
                return

            # Métrica integrada como texto (esto lo mantenemos)
            for metric, data in metrics.items():
                if isinstance(data, dict) and "value" in data:
                    value = round(data["value"], 4)
                    st.markdown(
                        f"El **{metric.upper()}** es **{value}**, "
                        f"lo que se interpreta de la siguiente forma:"
                    )

            if explanation:
                st.markdown(explanation)

            with st.expander("🔧 Detalle técnico"):
                st.json(metrics)

            if meta:
                with st.expander("📎 Metadatos"):
                    st.json(meta)

def render_assistant_text(answer: str):
    col_left, col_right = st.columns([4, 1])
    with col_left:
        with st.container(border=True):
            st.markdown("🤖 **Asistente**")
            st.markdown(answer)

def render_assistant_message(content):
    if not isinstance(content, dict):
        render_assistant_text(str(content))
        return

    if not content.get("ok", True):
        col_left, _ = st.columns([5, 2])
        with col_left:
            with st.container(border=True):
                st.markdown("🤖 **Asistente**")
                st.error(content.get("error", "Error desconocido"))
                st.json(content)
        return

    # SOLO screening si hay métricas reales
    if (
        isinstance(content.get("metrics"), dict)
        and content["metrics"]
    ):
        render_screening_result(content)

    # RAG textual
    elif "answer" in content:
        render_assistant_text(content["answer"])

    else:
        render_assistant_text(str(content))

# ==========================================================
# MAIN APP
# ==========================================================
def run_app():
    st.set_page_config(page_title="Agentic Screening RAG", layout="wide")

    # ------------------------------------------------------
    # HEADER
    # ------------------------------------------------------
    st.title("Screening de Licitaciones – Agentic RAG")

    st.markdown("""
    Sistema de análisis de licitaciones basado en:
    - RAG documental
    - Extracción numérica de ofertas
    - Cálculo automático de métricas de screening
    - Arquitectura agentic desacoplada
    """)

    st.divider()

    # ======================================================
    # INITIALIZATION
    # ======================================================
    if "embedder" not in st.session_state:
        st.session_state.embedder = Embedder("intfloat/e5-large-v2")

    if "vectorstore" not in st.session_state:
        os.makedirs("data/chroma", exist_ok=True)
        session_id = str(int(time.time()))
        collection_name = f"rag_session_{session_id}"

        st.session_state.vectorstore = ChromaVectorStore(
            persist_directory="data/chroma",
            collection_name=collection_name
        )

        st.info(f"Colección temporal creada: {collection_name}")

    if "agentic" not in st.session_state:
        st.session_state.agentic = initialize_screening_agentic(
            st.session_state.embedder,
            st.session_state.vectorstore
        )

    if "planner" not in st.session_state:
        st.session_state.planner = st.session_state.agentic["planner"]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ======================================================
    # DOCUMENT INGESTION
    # ======================================================
    st.header("Subida e indexación de documentación")

    uploaded_files = st.file_uploader(
        "Sube documentos (PDF, TXT, DOCX):",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Procesar e indexar"):
        with st.spinner("Procesando documentos..."):
            for file in uploaded_files:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.read())

                docs = process_file(temp_path)
                docs = st.session_state.embedder.embed_documents(docs)
                st.session_state.vectorstore.add_documents(docs)

                os.remove(temp_path)

        st.success("Documentación indexada correctamente.")

    # ======================================================
    # QUERY INPUT
    # ======================================================
    st.header("Consulta")

    query = st.text_input(
        "Introduce una consulta (ej: Calcula CV, RD y KSTEST para esta licitación)"
    )

    if st.button("Ejecutar") and query.strip():
        with st.spinner("El agente está analizando la licitación..."):
            result = st.session_state.planner.run(query)

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result
        })

        st.rerun()

    st.divider()

    # ======================================================
    # CHAT FLOW
    # ======================================================
    st.header("Conversación")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            content = msg["content"]
            if (
                isinstance(content, dict)
                and isinstance(content.get("metrics"), dict)
                and content["metrics"]
            ):
                render_screening_result(content)
            elif isinstance(content, dict) and "answer" in content:
                render_assistant_text(content["answer"])

# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    run_app()
