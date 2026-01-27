# vectorstore/chroma_store.py
import chromadb
from typing import List, Dict, Any
import uuid


class ChromaVectorStore:
    """
    Almacén vectorial persistente en ChromaDB.
    Compatible con fragmentos de texto y visuales (multimodales) sin depender de etiquetas fijas.
    Incluye saneamiento automático y metadatos generalistas.
    """

    def __init__(self, persist_directory: str = "data/chroma", collection_name: str = "rag_docs_general"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Inicialización de cliente persistente
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._get_or_create_collection(self.collection_name)

        print(f"[CHROMA] ✅ Conectado a la base persistente: {self.persist_directory}")
        print(f"[CHROMA] Colección activa: {self.collection.name}")

    # -------------------------------------------------------------------------
    # 1. CREACIÓN O RECUPERACIÓN DE COLECCIÓN
    # -------------------------------------------------------------------------
    def _get_or_create_collection(self, name: str):
        existing = [c.name for c in self.client.list_collections()]
        if name in existing:
            print(f"[CHROMA] Usando colección existente: {name}")
            return self.client.get_collection(name)
        print(f"[CHROMA] Creando nueva colección: {name}")
        return self.client.create_collection(name)

    # -------------------------------------------------------------------------
    # 2. SANITIZADOR DE METADATOS
    # -------------------------------------------------------------------------
    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Asegura que todos los valores sean válidos para Chroma (no None, no objetos)."""
        clean = {}
        for k, v in meta.items():
            if v is None:
                clean[k] = "desconocido"
            elif isinstance(v, (int, float, bool, str)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    # -------------------------------------------------------------------------
    # 3. AÑADIR DOCUMENTOS / FRAGMENTOS
    # -------------------------------------------------------------------------
    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        Agrega fragmentos con embeddings y metadatos al índice.
        Espera docs con: content, embedding, source, type, page (opcional).
        """
        if not docs:
            print("[CHROMA] ⚠️ Lista vacía: no se indexa nada.")
            return

        # IDs únicos: hash si existe, sino UUID
        ids = [
            d.get("hash") or f"{d.get('source', 'doc')}_{uuid.uuid4().hex[:8]}"
            for d in docs
        ]
        texts = [d["content"] for d in docs]
        embeddings = [d["embedding"] for d in docs]

        metadatas = []
        for d in docs:
            meta = {
                "source": d.get("source", "desconocido"),
                "type": d.get("type", "text"),
                "page": d.get("page", "N/A"),
            }
            metadatas.append(self._sanitize_metadata(meta))

        print(f"[CHROMA] Indexando {len(docs)} fragmentos (textuales y/o visuales)...")
        print(f"[CHROMA] Ejemplo de texto embebido:\n{texts[0][:250].replace(chr(10), ' ')}")

        # Inserción segura
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total = self.collection.count()
        print(f"[CHROMA] Total actual en colección '{self.collection.name}': {total}")

    # -------------------------------------------------------------------------
    # 4. CONSULTA DE DOCUMENTOS
    # -------------------------------------------------------------------------
    def query(self, query_embedding: List[float], query_text: str = "", top_k: int = 5, filter_type: str = None):
        """
        Recupera fragmentos más similares al embedding de consulta.
        Puede aplicar un filtro explícito por tipo ("text", "figure") si se desea.
        """
        where_filter = None
        if filter_type:
            where_filter = {"type": filter_type}
            print(f"[CHROMA] 🔎 Filtro manual: tipo = {filter_type}")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        if not results.get("documents") or not results["documents"][0]:
            print("[CHROMA] ⚠️ No se recuperaron fragmentos relevantes.")
            return {"documents": [], "sources": [], "metadatas": []}

        documents = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0]
        sources = [m.get("source", "desconocido") for m in metadatas]

        print(f"[CHROMA] Recuperados {len(documents)} fragmentos.")
        if sources:
            print(f"[CHROMA] Fuente top-1: {sources[0]}")
        print(f"[CHROMA] Ejemplo top-1:\n{documents[0][:400].replace(chr(10), ' ')}\n")

        return {"documents": documents, "sources": sources, "metadatas": metadatas}

    # -------------------------------------------------------------------------
    # 5. UTILIDADES
    # -------------------------------------------------------------------------
    def list_collections(self):
        """Lista las colecciones disponibles en la base persistente."""
        return [c.name for c in self.client.list_collections()]
    
    def reset_collection(self):
        """Elimina la colección actual y crea una nueva vacía."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)
        print(f"[CHROMA] Colección '{self.collection_name}' reiniciada.")
