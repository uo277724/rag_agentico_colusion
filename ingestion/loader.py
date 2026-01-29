import os
import fitz  # PyMuPDF
import docx
from ingestion.img_processor import ImageProcessor
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]

MAX_CHARS = 1000
OVERLAP_CHARS = 200


# --------------------------------------------------
# 1. Lectura layout-aware
# --------------------------------------------------
def _load_pdf_blocks(file_path: str):
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

    return sorted(blocks, key=lambda b: (b["page"], b["y0"], b["x0"]))


def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# --------------------------------------------------
# 2. Chunking estructural con solapamiento
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

        cursor = end - overlap_chars
        if cursor < 0:
            cursor = 0

    return chunks


# --------------------------------------------------
# 3. Pipeline principal
# --------------------------------------------------
def process_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)

    all_chunks = []

    if ext == ".pdf":
        blocks = _load_pdf_blocks(file_path)

        pages = {}
        for b in blocks:
            pages.setdefault(b["page"], []).append(b["text"])

        for page, texts in pages.items():
            page_text = "\n".join(texts)
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

        image_chunks = [{
            "type": "figure",
            "source": base_name,
            "page": img["page"],
            "content": img["description"]
        } for img in images]

        all_chunks.extend(image_chunks)

    elif ext == ".txt":
        raw = _load_txt(file_path)
        chunks = _chunk_with_overlap(
            raw,
            max_chars=MAX_CHARS,
            overlap_chars=OVERLAP_CHARS
        )

        for c in chunks:
            all_chunks.append({
                "type": "otro",
                "source": base_name,
                "content": c
            })

    elif ext == ".docx":
        raw = _load_docx(file_path)
        chunks = _chunk_with_overlap(
            raw,
            max_chars=MAX_CHARS,
            overlap_chars=OVERLAP_CHARS
        )

        for c in chunks:
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
