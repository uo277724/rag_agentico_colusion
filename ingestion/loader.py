# ingestion/loader.py
import os
import re
import fitz  # PyMuPDF
import docx
from ingestion.img_processor import ImageProcessor


SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]


# ------------------------
# 1. Lectura genérica
# ------------------------
def _load_pdf_blocks(file_path: str):
    """
    Extrae texto de cada página del PDF conservando posiciones (layout-aware).
    Devuelve lista de bloques con coordenadas.
    """
    blocks = []
    with fitz.open(file_path) as doc:
        for page_index, page in enumerate(doc):
            for (x0, y0, x1, y1, text, block_no, block_type) in page.get_text("blocks"):
                text = text.strip()
                if not text:
                    continue
                blocks.append({
                    "page": page_index + 1,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": text
                })

    # Ordenar por posición visual (de arriba a abajo, izquierda a derecha)
    blocks = sorted(blocks, key=lambda b: (b["page"], round(b["y0"]), round(b["x0"])))
    return blocks


def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


# ------------------------
# 2. Limpieza neutral
# ------------------------
def _clean_text(text: str) -> str:
    """
    Limpieza genérica de artefactos comunes a PDFs o documentos técnicos.
    No elimina términos cortos ni busca palabras específicas.
    """
    text = text.replace("\r", "\n")

    # Quitar numeraciones y encabezados/pies comunes
    text = re.sub(r"(^|\n)\s*\d+\s*/\s*\d+\s*(\n|$)", " ", text)
    text = re.sub(r"(^|\n)\s*Página\s*\d+(\s|$)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(^|\n)\s*\d+\s*$", " ", text)

    # Normalizar saltos y espacios
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    # Sustituir secuencias de puntos o tabulaciones
    text = re.sub(r"(\.{3,}|\t+)", " ", text)

    # Mantener todas las líneas con algo de contenido significativo
    lines = [
        ln.strip()
        for ln in text.split("\n")
        if len(re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúÑñ0-9]", "", ln)) > 2
    ]

    return "\n".join(lines).strip()


# ------------------------
# 3. Split semántico robusto
# ------------------------
def _split_semantic(text: str, target_words: int = 350, overlap_words: int = 50):
    """
    Segmenta por párrafos naturales y mantiene contexto con solapamiento.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current, word_count = [], [], 0

    for p in paragraphs:
        words = p.split()
        if word_count + len(words) > target_words and current:
            chunks.append(" ".join(current))
            overlap = " ".join(" ".join(current).split()[-overlap_words:])
            current = [overlap, p]
            word_count = len(overlap.split()) + len(words)
        else:
            current.append(p)
            word_count += len(words)

    if current:
        chunks.append(" ".join(current))

    chunks = [c for c in chunks if len(c.split()) > 20]

    avg_words = int(sum(len(c.split()) for c in chunks) / len(chunks)) if chunks else 0
    print(f"[LOADER] Generados {len(chunks)} fragmentos (media: {avg_words} palabras)")
    return chunks


# ------------------------
# 4. Asociación imagen-contexto
# ------------------------
def _associate_image_with_context(image, text_blocks, context_window=400):
    """
    Crea un chunk multimodal combinando la descripción de la imagen
    con el texto circundante en la misma página (sin buscar palabras específicas).
    """
    page = image["page"]

    # Selecciona bloques de texto cercanos a la posición vertical de la imagen
    neighbors = [
        b["text"] for b in text_blocks
        if b["page"] == page and abs(b["y0"] - image["y0"]) < context_window
    ]

    local_context = " ".join(neighbors).strip()
    if len(local_context.split()) < 30:
        local_context = "Contexto insuficiente."

    content = (
        f"[Contexto en la página {page}]\n"
        f"{local_context}\n\n"
        f"[Descripción visual generada]\n"
        f"{image['description']}"
    )

    return {
        "page": page,
        "source": os.path.basename(image.get('source', '')),
        "type": "figure",
        "content": content,
        "hash": image.get("hash"),
        "index": image.get("index"),
    }


# ------------------------
# 4.5 Contextual Chunk Enrichment (CCE)
# ------------------------
def _contextualize_chunk(doc: dict, full_text: str, window: int = 300) -> dict:
    """
    Amplía un fragmento con contexto local (texto antes y después)
    para mejorar la representación semántica durante el embedding.
    """
    try:
        content = doc.get("content", "")
        if not content.strip():
            return doc

        start_idx = full_text.find(content[:100])
        if start_idx == -1:
            return doc

        end_idx = start_idx + len(content)
        start = max(0, start_idx - window)
        end = min(len(full_text), end_idx + window)
        local_context = full_text[start:end].strip()

        if local_context and len(local_context) > len(content):
            doc["content"] = local_context

        return doc
    except Exception as e:
        print(f"[CCE] Error contextualizando chunk: {e}")
        return doc


# ------------------------
# 5. Pipeline principal
# ------------------------
def process_file(file_path: str, target_words: int = 350, overlap_words: int = 50):
    """
    Procesa un archivo (PDF, TXT o DOCX) y devuelve lista de fragmentos listos para embeddings.
    Si es PDF, incluye texto, imágenes y contexto circundante combinados.
    """
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)

    if ext == ".pdf":
        print(f"[LOADER] Procesando PDF con layout y figuras: {file_path}")

        # 1. Texto
        text_blocks = _load_pdf_blocks(file_path)
        cleaned_text = _clean_text("\n".join(b["text"] for b in text_blocks))
        text_chunks = _split_semantic(cleaned_text, target_words, overlap_words)

        # 2. Imágenes
        img_processor = ImageProcessor()
        images = img_processor.process_pdf_images(file_path)

        # 3. Crear chunks multimodales combinando contexto e imagen
        multimodal_chunks = [
            _associate_image_with_context(img, text_blocks)
            for img in images
        ]

        # 4. Combinar ambos tipos de fragmentos
        text_chunks = [{"content": c, "source": base_name, "type": "text"} for c in text_chunks]
        all_chunks = text_chunks + multimodal_chunks

    elif ext == ".txt":
        raw_text = _load_txt(file_path)
        cleaned = _clean_text(raw_text)
        chunks = _split_semantic(cleaned, target_words, overlap_words)
        all_chunks = [{"content": c, "source": base_name, "type": "text"} for c in chunks]

    elif ext == ".docx":
        raw_text = _load_docx(file_path)
        cleaned = _clean_text(raw_text)
        chunks = _split_semantic(cleaned, target_words, overlap_words)
        all_chunks = [{"content": c, "source": base_name, "type": "text"} for c in chunks]

    else:
        raise ValueError(f"Formato no soportado: {ext}")

    # ------------------------
    # Contextual Chunk Enrichment (CCE)
    # ------------------------
    try:
        full_text = cleaned_text if ext == ".pdf" else cleaned
        enriched_chunks = []
        for doc in all_chunks:
            if doc.get("type") == "text":
                enriched = _contextualize_chunk(doc, full_text, window=300)
                enriched_chunks.append(enriched)
            else:
                enriched_chunks.append(doc)
        all_chunks = enriched_chunks
        print(f"[CCE] Enriquecidos {len([d for d in all_chunks if d['type']=='text'])} fragmentos con contexto local.")
    except Exception as e:
        print(f"[CCE] Error en el enriquecimiento contextual: {e}")

    # ------------------------
    # Reporte final
    # ------------------------
    print(f"[LOADER] Archivo procesado: {file_path}")
    print(f"[LOADER] Total de fragmentos generados: {len(all_chunks)}")
    if all_chunks:
        ejemplo = all_chunks[0]['content'][:400].replace("\n", " ")
        print(f"[LOADER] Ejemplo de fragmento:\n{ejemplo}")

    return all_chunks