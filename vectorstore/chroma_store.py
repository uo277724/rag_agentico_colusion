import chromadb
from typing import List, Dict, Any, Optional
import uuid


class ChromaVectorStore:
    """
    Almacén vectorial persistente consciente de estructura documental.
    Soporta múltiples embeddings por documento según el modo de uso.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_docs_general",
        embedding_mode: str = "index",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_mode = embedding_mode

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._get_or_create_collection(self.collection_name)

        print(f"[CHROMA] Conectado a: {self.persist_directory}")
        print(f"[CHROMA] Colección activa: {self.collection.name}")
        print(f"[CHROMA] Modo de embedding: {self.embedding_mode}")

    # --------------------------------------------------
    # Colección
    # --------------------------------------------------
    def _get_or_create_collection(self, name: str):
        existing = {c.name for c in self.client.list_collections()}
        if name in existing:
            return self.client.get_collection(name)
        return self.client.create_collection(name)

    # --------------------------------------------------
    # Sanitización
    # --------------------------------------------------
    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}
        for k, v in meta.items():
            if v is None:
                clean[k] = "unknown"
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    # --------------------------------------------------
    # Indexado
    # --------------------------------------------------
    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        Espera documentos con:
        - content
        - embedding_<mode>
        - type
        - source
        - page (opcional)
        - confidence (opcional)
        """

        if not docs:
            print("[CHROMA] Nada que indexar.")
            return

        embedding_key = f"embedding_{self.embedding_mode}"

        valid_docs = [d for d in docs if embedding_key in d]
        if not valid_docs:
            print(f"[CHROMA] No hay embeddings '{embedding_key}'.")
            return

        ids, texts, embeddings, metadatas = [], [], [], []

        for d in valid_docs:
            doc_id = d.get("hash") or f"{d.get('source','doc')}_{uuid.uuid4().hex[:8]}"
            ids.append(doc_id)

            texts.append(d["content"])
            embeddings.append(d[embedding_key])

            meta = {
                "source": d.get("source", "unknown"),
                "type": d.get("type", "unknown"),
                "page": d.get("page"),
                "confidence": d.get("confidence", 1.0),
            }
            metadatas.append(self._sanitize_metadata(meta))

        print(f"[CHROMA] Indexando {len(ids)} documentos (modo={self.embedding_mode})")

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[CHROMA] Total en colección: {self.collection.count()}")

    # --------------------------------------------------
    # Consulta
    # --------------------------------------------------
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ):
        """
        Recupera documentos similares respetando estructura.
        """

        where = {}
        if filter_types:
            where["type"] = {"$in": filter_types}
        if min_confidence > 0:
            where["confidence"] = {"$gte": min_confidence}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            print("[CHROMA] Sin resultados.")
            return []

        print(f"[CHROMA] Recuperados {len(docs)} documentos.")
        return [
            {
                "content": d,
                "metadata": m,
            }
            for d, m in zip(docs, metas)
        ]

    # --------------------------------------------------
    # Utilidades
    # --------------------------------------------------
    def reset_collection(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)
        print(f"[CHROMA] Colección '{self.collection_name}' reiniciada.")

    def list_collections(self):
        return [c.name for c in self.client.list_collections()]
