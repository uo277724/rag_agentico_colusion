import os
import time
import uuid
import streamlit as st

# ==========================================================
# INGESTION + EMBEDDINGS
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
                    <div style="font-weight: 600;">🧑 User</div>
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
            st.markdown("🤖 **Assistant**")

            if not metrics:
                st.warning("No metrics were returned.")
                return

            if explanation:
                st.markdown(explanation)

            with st.expander("🔧 Technical details"):
                st.json(metrics)

            if meta:
                with st.expander("📎 Metadata"):
                    st.json(meta)

def render_assistant_text(answer: str):
    col_left, col_right = st.columns([4, 1])
    with col_left:
        with st.container(border=True):
            st.markdown("🤖 **Assistant**")
            st.markdown(answer)

def render_assistant_message(content):
    if not isinstance(content, dict):
        render_assistant_text(str(content))
        return

    if not content.get("ok", True):
        col_left, _ = st.columns([5, 2])
        with col_left:
            with st.container(border=True):
                st.markdown("🤖 **Assistant**")
                st.error(content.get("error", "Unknown error"))
                st.json(content)
        return

    # ALWAYS render textual answer if present
    if "answer" in content and content["answer"]:
        render_assistant_text(content["answer"])

    # Then render metrics if they exist
    if isinstance(content.get("metrics"), dict) and content["metrics"]:
        render_screening_result(content)

    if not content.get("ok", True):
        col_left, _ = st.columns([5, 2])
        with col_left:
            with st.container(border=True):
                st.markdown("🤖 **Assistant**")
                st.error(content.get("error", "Unknown error"))
                st.json(content)
        return

    # SCREENING only if real metrics are present
    if (
        isinstance(content.get("metrics"), dict)
        and content["metrics"]
    ):
        render_screening_result(content)

    # Textual RAG
    elif "answer" in content:
        render_assistant_text(content["answer"])

    else:
        render_assistant_text(str(content))

# ==========================================================
# MAIN APP
# ==========================================================
def run_app():
    st.set_page_config(page_title="Agentic Screening RAG", layout="wide")

    with st.sidebar:
        st.header("Settings")
        st.session_state.use_graph_rag = st.checkbox(
            "GraphRAG mode",
            value=st.session_state.get("use_graph_rag", False),
        )

    # ------------------------------------------------------
    # HEADER
    # ------------------------------------------------------
    st.title("Tender Screening – Agentic RAG")

    st.markdown("""
    Tender analysis system based on:
    - Document-based RAG
    - Numerical extraction of bids
    - Automatic calculation of screening metrics
    - Decoupled agentic architecture
    """)

    st.divider()

    # ======================================================
    # INITIALIZATION
    # ======================================================
    if "embedder" not in st.session_state:
        st.session_state.embedder = Embedder("intfloat/e5-large-v2")

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())

    if "vectorstore" not in st.session_state:
        os.makedirs("data/chroma", exist_ok=True)
        session_id = str(int(time.time()))
        collection_name = f"rag_session_{session_id}"

        st.session_state.vectorstore = ChromaVectorStore(
            persist_directory="data/chroma",
            collection_name=collection_name
        )

        st.info(f"Temporary collection created: {collection_name}")

    if "agentic" not in st.session_state:
        st.session_state.agentic = initialize_screening_agentic(
            st.session_state.embedder,
            st.session_state.vectorstore
        )

    if "planner" not in st.session_state:
        st.session_state.planner = st.session_state.agentic["planner"]

    if "graph_store" not in st.session_state:
        st.session_state.graph_store = st.session_state.agentic.get("graph_store")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ======================================================
    # DOCUMENT INGESTION
    # ======================================================
    st.header("Document upload and indexing")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX):",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process and index"):
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.read())

                docs = process_file(temp_path)
                docs = st.session_state.embedder.embed_documents(docs)
                st.session_state.vectorstore.add_documents(docs)

                if st.session_state.graph_store:
                    st.session_state.graph_store.add_chunks(docs)

                os.remove(temp_path)

        st.success("Documentation indexed successfully.")

    # ======================================================
    # QUERY INPUT
    # ======================================================
    st.header("Query")

    query = st.text_input(
        "Enter a query (e.g.: Calculate CV, RD and KSTEST for this tender)"
    )

    if st.button("Run") and query.strip():
        with st.spinner("The agent is analyzing the tender..."):
            result = st.session_state.planner.run(
                query=query,
                conversation_id=st.session_state.conversation_id,
                use_graph_rag=st.session_state.get("use_graph_rag", False),
            )

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
    st.header("Conversation")

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
