# embeddings/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import re


class Embedder:
    """
    Genera embeddings de alta fidelidad para texto técnico y multimodal.
    Totalmente generalista: no depende de etiquetas o términos específicos.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        normalize_embeddings: bool = True,
        batch_size: int = 16,
        device: str = None,
    ):
        print(f"[EMBEDDER] Cargando modelo: {model_name}")

        # Carga automática en GPU si está disponible
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = SentenceTransformer(model_name, device=self.device)
        self.normalize = normalize_embeddings
        self.batch_size = batch_size

        dim = self.model.get_sentence_embedding_dimension()
        print(f"[EMBEDDER] Modelo cargado correctamente (dim={dim}) en dispositivo: {self.device}")

    # -------------------------------------------------------------------------
    # 1. Limpieza ligera (sin normalizar semántica)
    # -------------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """Limpieza neutra: elimina ruido y normaliza espacios."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\s{2,}", " ", text)
        text = text.replace("\r", " ").replace("\n", " ").strip()
        return text

    # -------------------------------------------------------------------------
    # 2. Enriquecimiento contextual
    # -------------------------------------------------------------------------
    def _enrich_with_metadata(self, doc: dict) -> str:
        """
        Integra metadatos estructurales antes del embedding,
        sin modificar el contenido semántico del fragmento.
        """
        content = self._clean_text(doc.get("content", ""))

        meta_parts = []
        if doc.get("type"):
            meta_parts.append(f"Tipo: {doc['type'].capitalize()}")
        if doc.get("page"):
            meta_parts.append(f"Página: {doc['page']}")
        if doc.get("source"):
            meta_parts.append(f"Fuente: {doc['source']}")

        # Prefijo informativo (sin sesgo semántico)
        if meta_parts:
            prefix = " | ".join(meta_parts)
            enriched = f"[{prefix}] {content}"
        else:
            enriched = content

        return enriched.strip()

    # -------------------------------------------------------------------------
    # 3. Embeddings simples (texto directo)
    # -------------------------------------------------------------------------
    def embed_texts(self, texts):
        """Genera embeddings para una lista de textos simples."""
        if not texts:
            print("[EMBEDDER] Lista vacía: no hay textos para procesar.")
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )

        print(f"[EMBEDDER] {len(embeddings)} embeddings generados (dim={embeddings.shape[1]})")
        return np.array(embeddings)

    # -------------------------------------------------------------------------
    # 4. Embeddings con metadatos (documentos)
    # -------------------------------------------------------------------------
    def embed_documents(self, docs):
        """
        Recibe una lista de dicts [{"content":..., "source":..., "type":...}].
        Añade 'embedding' a cada entrada.
        """
        if not docs:
            print("[EMBEDDER] No se recibieron documentos.")
            return []

        valid_docs = [d for d in docs if d.get("content") and d["content"].strip()]
        skipped = len(docs) - len(valid_docs)
        if skipped > 0:
            print(f"[EMBEDDER] {skipped} fragmentos vacíos omitidos.")

        if not valid_docs:
            print("[EMBEDDER] No hay fragmentos válidos para generar embeddings.")
            return []

        enriched_texts = [self._enrich_with_metadata(d) for d in valid_docs]
        print(f"[EMBEDDER] Generando embeddings para {len(enriched_texts)} fragmentos...")

        embeddings = self.model.encode(
            enriched_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )

        dim = embeddings.shape[1]
        print(f"[EMBEDDER] Embeddings generados (dimensión: {dim})")

        for i, doc in enumerate(valid_docs):
            doc["embedding"] = embeddings[i].tolist()

        print(f"[EMBEDDER] Proceso completado: {len(valid_docs)} fragmentos embebidos correctamente.")
        return valid_docs