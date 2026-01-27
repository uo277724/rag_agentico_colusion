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
def render_screening_result(result: dict):
    st.subheader("📊 Resultados de Screening")

    metrics = result.get("metrics", {})
    meta = result.get("meta", {})
    explanation = result.get("explanation")

    if not metrics:
        st.warning("No se han devuelto métricas.")
        return

    # -----------------------------
    # Métricas numéricas
    # -----------------------------
    cols = st.columns(len(metrics))

    for col, (metric, data) in zip(cols, metrics.items()):
        with col:
            if isinstance(data, dict) and "value" in data:
                st.metric(
                    label=metric.upper(),
                    value=round(data["value"], 4)
                )
            else:
                st.metric(label=metric.upper(), value="N/A")

    # -----------------------------
    # Explicación en lenguaje natural
    # -----------------------------
    if explanation:
        st.subheader("🧠 Interpretación del resultado")
        st.markdown(explanation)

    # -----------------------------
    # Detalle técnico
    # -----------------------------
    with st.expander("📐 Detalle técnico"):
        st.json(metrics)

    if meta:
        with st.expander("📎 Metadatos"):
            st.json(meta)


# ==========================================================
# MAIN APP
# ==========================================================
def run_app():
    st.set_page_config(page_title="Agentic Screening RAG", layout="wide")

    # ------------------------------------------------------
    # HEADER
    # ------------------------------------------------------
    if os.path.exists("./data/itur-web-1.webp"):
        st.image("./data/itur-web-1.webp", width=400)

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

    # ---- Embedder ----
    if "embedder" not in st.session_state:
        st.session_state.embedder = Embedder("intfloat/e5-large-v2")

    # ---- Vectorstore ----
    if "vectorstore" not in st.session_state:
        os.makedirs("data/chroma", exist_ok=True)
        session_id = str(int(time.time()))
        collection_name = f"rag_session_{session_id}"

        st.session_state.vectorstore = ChromaVectorStore(
            persist_directory="data/chroma",
            collection_name=collection_name
        )

        st.info(f"Colección temporal creada: {collection_name}")

    # ---- Agentic Screening ----
    if "agentic" not in st.session_state:
        st.session_state.agentic = initialize_screening_agentic(
            st.session_state.embedder,
            st.session_state.vectorstore
        )

    if "planner" not in st.session_state:
        st.session_state.planner = st.session_state.agentic["planner"]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.divider()

    # ======================================================
    # DOCUMENT INGESTION
    # ======================================================
    st.header("1️⃣ Subida e indexación de documentación")

    uploaded_files = st.file_uploader(
        "Sube documentos (PDF, TXT, DOCX):",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 Procesar e indexar"):
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

    st.divider()

    # ======================================================
    # QUERY
    # ======================================================
    st.header("2️⃣ Consulta")

    query = st.text_input(
        "Introduce una consulta (ej: 'Calcula CV, RD y KSTEST para esta licitación'):"
    )

    submit_query = st.button("🚀 Ejecutar")

    if submit_query and query.strip():
        with st.spinner("El agente está analizando la licitación…"):
            result = st.session_state.planner.run(query)

        if not result.get("ok"):
            st.error(f"❌ Error: {result.get('error')}")
            st.json(result)
        else:
            if "answer" in result:
                st.subheader("📄 Respuesta documental")
                st.markdown(result["answer"])
            elif "metrics" in result:
                render_screening_result(result)
            else:
                st.warning("Resultado devuelto en formato no reconocido")
                st.json(result)

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
    # LAST RESPONSE
    # ======================================================
    if st.session_state.chat_history:
        last_user = next(
            (m for m in reversed(st.session_state.chat_history) if m["role"] == "user"),
            None,
        )
        last_assistant = next(
            (m for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
            None,
        )

        if last_user and last_assistant:
            st.markdown("## 🟦 Última interacción")
            st.markdown(f"**👤 Usuario:** {last_user['content']}")

            content = last_assistant["content"]
            if isinstance(content, dict) and "metrics" in content:
                render_screening_result(content)
            elif isinstance(content, dict) and "answer" in content:
                st.markdown(content["answer"])

    st.divider()

    # ======================================================
    # HISTORY
    # ======================================================
    st.header("📜 Historial completo")

    for msg in st.session_state.chat_history[:-2]:
        if msg["role"] == "user":
            st.markdown(f"**👤 Usuario:** {msg['content']}")
        else:
            content = msg["content"]
            if isinstance(content, dict) and "metrics" in content:
                render_screening_result(content)
            elif isinstance(content, dict) and "answer" in content:
                st.markdown(content["answer"])

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🧹 Limpiar historial"):
            st.session_state.chat_history.clear()
            st.rerun()


# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    run_app()
