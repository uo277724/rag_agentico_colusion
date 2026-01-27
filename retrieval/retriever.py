# retrieval/retriever.py
from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaVectorStore


class Retriever:
    """
    Recupera contexto relevante desde el vector store usando embeddings semánticos.
    Compatible con documentos multimodales (texto + visuales).
    No depende de palabras o formatos específicos (p. ej., 'figura', 'ilustración', etc.).
    """

    def __init__(self, embedder: Embedder, vectorstore: ChromaVectorStore, top_k: int = 30):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.top_k = top_k

    # -------------------------------------------------------------------------
    # Recuperación principal
    # -------------------------------------------------------------------------
    def retrieve(self, query: str, filter_type: str = None):
        """
        Recupera fragmentos relevantes para una consulta dada.
        Puede aplicar un filtro opcional: filter_type ∈ {"text", "figure"}.
        """
        print("\n[RETRIEVER] ======================")
        print(f"[RETRIEVER] Consulta recibida: {query}")

        # 1. Generar embedding de la consulta
        query_embedding = self.embedder.embed_texts([query])[0]
        print(f"[RETRIEVER] Dimensión del embedding: {len(query_embedding)}")

        # 2. Ejecutar consulta vectorial
        results = self.vectorstore.query(
            query_embedding=query_embedding,
            query_text=query,   # solo para logging interno
            top_k=self.top_k,
            filter_type=filter_type,
        )

        documents = results.get("documents", [])
        sources = results.get("sources", [])
        metadatas = results.get("metadatas", [])
        num_docs = len(documents)

        print(f"[RETRIEVER] Fragmentos recuperados: {num_docs}")

        if num_docs == 0:
            print("[RETRIEVER] ⚠️ No se encontraron fragmentos relevantes.")
            print("[RETRIEVER] ======================\n")
            return {
                "context": "",
                "sources": [],
                "metadatas": [],
                "query_embedding": query_embedding.tolist(),
            }

        # 3. Vista previa (solo primeros 3)
        for i in range(min(3, num_docs)):
            meta = metadatas[i]
            preview = documents[i][:400].replace("\n", " ")
            print(f"[RETRIEVER] Fragmento {i+1} — Tipo: {meta.get('type', 'N/A')} | Página: {meta.get('page', 'N/A')}")
            print(f"[RETRIEVER] Fuente: {meta.get('source', 'N/A')}")
            print(f"[RETRIEVER] Contenido: {preview}")
            print("---")

        # 4. Construir contexto concatenado
        context = "\n---\n".join(documents[:num_docs])
        print(f"[RETRIEVER] Longitud total del contexto: {len(context)} caracteres")
        print(f"[RETRIEVER] Fuentes únicas: {set(sources)}")
        print("[RETRIEVER] ======================\n")

        return {
            "context": context,
            "sources": sources[:num_docs],
            "metadatas": metadatas[:num_docs],
            "query_embedding": query_embedding.tolist(),
        }