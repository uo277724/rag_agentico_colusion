from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import re


class Embedder:
    """
    Genera embeddings semánticos preservando estructura documental.
    El embedder no decide relevancia: conserva información funcional
    para que el RAG y los agentes razonen correctamente.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        normalize_embeddings: bool = True,
        batch_size: int = 16,
        device: str = None,
    ):
        print(f"[EMBEDDER] Cargando modelo: {model_name}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = SentenceTransformer(model_name, device=self.device)
        self.normalize = normalize_embeddings
        self.batch_size = batch_size

        dim = self.model.get_sentence_embedding_dimension()
        print(f"[EMBEDDER] Modelo cargado (dim={dim}) en {self.device}")

    # --------------------------------------------------
    # Limpieza mínima (no semántica)
    # --------------------------------------------------
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\s{2,}", " ", text)
        return text.replace("\n", " ").strip()

    # --------------------------------------------------
    # Construcción de input para embedding
    # --------------------------------------------------
    def _build_embedding_input(self, doc: dict, mode: str) -> str:
        """
        mode:
        - index: recuperación RAG
        - extract: extracción de bids
        - explain: explicaciones / auditoría
        """

        content = self._clean_text(doc.get("content", ""))

        if mode == "index":
            # Preservar tipo de forma explícita y consistente
            prefix = f"DOCUMENT TYPE: {doc.get('type', 'unknown')}. "
            return prefix + content

        if mode == "extract":
            # Máxima pureza semántica, mínimo ruido
            return content

        if mode == "explain":
            # Contexto rico para trazabilidad
            meta = []
            if doc.get("type"):
                meta.append(f"Tipo: {doc['type']}")
            if doc.get("page"):
                meta.append(f"Página: {doc['page']}")
            if doc.get("source"):
                meta.append(f"Fuente: {doc['source']}")

            header = " | ".join(meta)
            return f"[{header}] {content}" if header else content

        return content

    # --------------------------------------------------
    # Embeddings genéricos
    # --------------------------------------------------
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)

    # --------------------------------------------------
    # Embeddings documentales con modo
    # --------------------------------------------------
    def embed_documents(self, docs: list[dict], mode: str = "index") -> list[dict]:
        """
        Añade embeddings a los documentos.
        mode controla cómo se construye el embedding.
        """

        valid_docs = [d for d in docs if d.get("content", "").strip()]
        if not valid_docs:
            print("[EMBEDDER] No hay documentos válidos.")
            return []

        inputs = [
            self._build_embedding_input(d, mode=mode)
            for d in valid_docs
        ]

        print(f"[EMBEDDER] Generando embeddings ({mode}) para {len(inputs)} documentos...")

        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        for doc, emb in zip(valid_docs, embeddings):
            doc[f"embedding_{mode}"] = emb.tolist()

        return valid_docs
