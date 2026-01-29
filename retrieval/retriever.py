# retrieval/retriever.py

from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaVectorStore
from typing import List, Dict, Any


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vectorstore: ChromaVectorStore,
        top_k_primary: int = 8,
        top_k_secondary: int = 20,
        min_confidence: float = 0.0,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.top_k_primary = top_k_primary
        self.top_k_secondary = top_k_secondary
        self.min_confidence = min_confidence

    def _embed_query(self, query: str) -> List[float]:
        query_doc = {"content": query}
        embedded = self.embedder.embed_documents(
            [query_doc],
            mode="index"
        )
        return embedded[0]["embedding_index"]

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

    def retrieve(self, query: str):
        print("\n[RETRIEVER] ======================")
        print(f"[RETRIEVER] Consulta: {query}")

        query_embedding = self._embed_query(query)
        print(f"[RETRIEVER] Embedding generado (dim={len(query_embedding)})")

        raw_result = self._retrieve_once(
            query_embedding=query_embedding,
            top_k=self.top_k_secondary,
        )

        if isinstance(raw_result, dict):
            documents = raw_result.get("documents", [])
        elif isinstance(raw_result, list):
            documents = raw_result
        else:
            documents = []

        print(f"[RETRIEVER] Normalized documents count: {len(documents)}")
        print("[RETRIEVER] ======================\n")

        return documents
