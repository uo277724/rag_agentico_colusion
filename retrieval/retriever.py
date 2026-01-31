# retrieval/retriever.py

from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaVectorStore
from typing import List, Dict, Any, Optional


class Retriever:
    """
    Responsable de:
    - Embedear la consulta
    - Recuperar top-K documentos del vectorstore
    - Anotar opcionalmente con tipado semántico perezoso

    NO decide relevancia.
    NO filtra por tipo.
    NO aplica lógica de negocio.
    """

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: ChromaVectorStore,
        top_k_primary: int = 8,
        top_k_secondary: int = 20,
        min_confidence: float = 0.0,
        lazy_typer: Optional[Any] = None,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.top_k_primary = top_k_primary
        self.top_k_secondary = top_k_secondary
        self.min_confidence = min_confidence
        self.lazy_typer = lazy_typer

    # --------------------------------------------------
    # Embedding de consulta
    # --------------------------------------------------
    def _embed_query(self, query: str) -> List[float]:
        query_doc = {"content": query}
        embedded = self.embedder.embed_documents(
            [query_doc],
            mode="index"
        )
        return embedded[0]["embedding_index"]

    # --------------------------------------------------
    # Llamada al vectorstore
    # --------------------------------------------------
    def _retrieve_once(
        self,
        query_embedding: List[float],
        top_k: int,
    ):
        print("[RETRIEVER] Calling vectorstore.query(...)")

        result = self.vectorstore.query(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        print("[RETRIEVER] Raw vectorstore.query result type:", type(result))
        print("[RETRIEVER] Raw vectorstore.query result value:", result)

        return result

    # --------------------------------------------------
    # API pública
    # --------------------------------------------------
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        print("\n[RETRIEVER] ======================")
        print(f"[RETRIEVER] Consulta: {query}")

        query_embedding = self._embed_query(query)
        print(f"[RETRIEVER] Embedding generado (dim={len(query_embedding)})")

        raw_result = self._retrieve_once(
            query_embedding=query_embedding,
            top_k=self.top_k_secondary,
        )

        # Normalización defensiva
        if isinstance(raw_result, dict):
            documents = raw_result.get("documents", []) or []
        elif isinstance(raw_result, list):
            documents = raw_result
        else:
            documents = []

        print(f"[RETRIEVER] Normalized documents count: {len(documents)}")

        # --------------------------------------------------
        # Tipado semántico perezoso (ENRIQUECIMIENTO)
        # --------------------------------------------------
        if self.lazy_typer and documents:
            print("[RETRIEVER] Applying lazy semantic typing...")
            try:
                documents = self.lazy_typer.annotate(documents)
            except Exception as e:
                # Fallback silencioso: el retriever NUNCA debe fallar por el typer
                print(f"[RETRIEVER] Lazy typing failed: {e}")

        print("[RETRIEVER] ======================\n")
        return documents
