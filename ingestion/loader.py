import os
import fitz  # PyMuPDF
import docx
from ingestion.img_processor import ImageProcessor
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]

MAX_CHARS = 1000
OVERLAP_CHARS = 200


# --------------------------------------------------
# 1. Lectura layout-aware REAL
# --------------------------------------------------
def _load_pdf_blocks(file_path: str) -> List[Dict]:
    blocks = []
    with fitz.open(file_path) as doc:
        for page_index, page in enumerate(doc):
            for (x0, y0, x1, y1, text, *_ ) in page.get_text("blocks"):
                text = text.strip()
                if text:
                    blocks.append({
                        "page": page_index + 1,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "text": text
                    })
    return blocks


# --------------------------------------------------
# 2. Detección geométrica de columnas
# --------------------------------------------------
def _split_columns(blocks: List[Dict]) -> List[List[Dict]]:
    xs = sorted(b["x0"] for b in blocks)
    if not xs:
        return [blocks]

    min_x, max_x = xs[0], xs[-1]
    width = max_x - min_x

    # Documento de una sola columna
    if width < 200:
        return [blocks]

    mid_x = min_x + width / 2

    left = []
    right = []

    for b in blocks:
        if b["x0"] < mid_x:
            left.append(b)
        else:
            right.append(b)

    return [left, right]


# --------------------------------------------------
# 3. Agrupación de bloques cercanos (listas)
# --------------------------------------------------
def _merge_close_blocks(blocks: List[Dict], y_gap=25, x_tolerance=20) -> List[str]:
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: b["y0"])
    merged = []
    current = blocks[0]["text"]
    prev = blocks[0]

    for b in blocks[1:]:
        same_column = abs(b["x0"] - prev["x0"]) < x_tolerance
        close_vertically = abs(b["y0"] - prev["y1"]) < y_gap

        if same_column and close_vertically:
            current += "\n" + b["text"]
        else:
            merged.append(current)
            current = b["text"]

        prev = b

    merged.append(current)
    return merged


# --------------------------------------------------
# 4. Chunking con solapamiento (al final)
# --------------------------------------------------
def _chunk_with_overlap(text: str, max_chars: int, overlap_chars: int):
    chunks = []
    cursor = 0
    length = len(text)

    while cursor < length:
        end = min(cursor + max_chars, length)
        chunk = text[cursor:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == length:
            break

        cursor = max(0, end - overlap_chars)

    return chunks


# --------------------------------------------------
# 5. Pipeline principal
# --------------------------------------------------
def process_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)

    all_chunks = []

    if ext == ".pdf":
        blocks = _load_pdf_blocks(file_path)

        pages = {}
        for b in blocks:
            pages.setdefault(b["page"], []).append(b)

        for page, page_blocks in pages.items():
            columns = _split_columns(page_blocks)

            ordered_text_blocks = []
            for col in columns:
                col_blocks = sorted(col, key=lambda b: b["y0"])
                merged = _merge_close_blocks(col_blocks)
                ordered_text_blocks.extend(merged)

            page_text = "\n\n".join(ordered_text_blocks)

            chunks = _chunk_with_overlap(
                page_text,
                max_chars=MAX_CHARS,
                overlap_chars=OVERLAP_CHARS
            )

            for c in chunks:
                all_chunks.append({
                    "type": "otro",
                    "source": base_name,
                    "page": page,
                    "content": c
                })

        # Imágenes (sin cambios)
        img_processor = ImageProcessor()
        images = img_processor.process_pdf_images(file_path)

        for img in images:
            all_chunks.append({
                "type": "figure",
                "source": base_name,
                "page": img["page"],
                "content": img["description"]
            })

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        for c in _chunk_with_overlap(raw, MAX_CHARS, OVERLAP_CHARS):
            all_chunks.append({
                "type": "otro",
                "source": base_name,
                "content": c
            })

    elif ext == ".docx":
        doc = docx.Document(file_path)
        raw = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        for c in _chunk_with_overlap(raw, MAX_CHARS, OVERLAP_CHARS):
            all_chunks.append({
                "type": "otro",
                "source": base_name,
                "content": c
            })

    else:
        raise ValueError(f"Formato no soportado: {ext}")

    print(f"[LOADER] {len(all_chunks)} chunks estructurales generados")
    if all_chunks:
        print("[LOADER] Ejemplo:")
        print(all_chunks[0]["type"], "→", all_chunks[0]["content"][:300])

    return all_chunks
