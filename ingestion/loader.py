import os
import fitz  # PyMuPDF
import docx
from openai import OpenAI
from ingestion.img_processor import ImageProcessor
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]


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
# 2. Segmentación semántica funcional (LLM)
# --------------------------------------------------
def _semantic_segment(text: str, model: str = "gpt-4.1-mini") -> list[dict]:
    """
    Divide el texto en unidades documentales funcionales:
    tablas económicas, descripciones, anexos, etc.
    No extrae datos, solo segmenta y tipifica.
    """

    client = OpenAI()

    prompt = """
    Segmenta el siguiente texto en unidades documentales funcionales coherentes.
    Para cada unidad indica:
    - type: tipo funcional (ej. tabla_economica, descripcion, anexo, presupuesto_base, otro)
    - content: texto completo de esa unidad
    - confidence: confianza de que sea una unidad autónoma (0–1)

    Reglas:
    - No extraigas precios ni valores.
    - No resumas.
    - No inventes contenido.
    - Cada unidad debe poder entenderse de forma autónoma.
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Eres un analista documental experto en contratación pública."},
            {"role": "user", "content": prompt + "\n\nTEXTO:\n" + text}
        ]
    )

    try:
        units = eval(response.choices[0].message.content)
        return [u for u in units if u.get("confidence", 0) >= 0.5]
    except Exception:
        return [{
            "type": "otro",
            "content": text,
            "confidence": 0.3
        }]


# --------------------------------------------------
# 3. Chunking condicionado por tipo
# --------------------------------------------------
def _chunk_by_type(units: list[dict], source: str) -> list[dict]:
    chunks = []

    for u in units:
        t = u["type"]
        content = u["content"].strip()

        if len(content.split()) < 20:
            continue

        chunks.append({
            "type": t,
            "source": source,
            "content": content
        })

    return chunks


# --------------------------------------------------
# 4. Pipeline principal
# --------------------------------------------------
def process_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)

    if ext == ".pdf":
        blocks = _load_pdf_blocks(file_path)

        # Agrupar por página para no mezclar contextos
        pages = {}
        for b in blocks:
            pages.setdefault(b["page"], []).append(b["text"])

        semantic_units = []
        for page, texts in pages.items():
            page_text = "\n".join(texts)
            semantic_units.extend(_semantic_segment(page_text))

        text_chunks = _chunk_by_type(semantic_units, base_name)

        # Imágenes (se mantienen igual, pero separadas)
        img_processor = ImageProcessor()
        images = img_processor.process_pdf_images(file_path)

        image_chunks = [{
            "type": "figure",
            "source": base_name,
            "page": img["page"],
            "content": img["description"]
        } for img in images]

        all_chunks = text_chunks + image_chunks

    elif ext == ".txt":
        raw = _load_txt(file_path)
        units = _semantic_segment(raw)
        all_chunks = _chunk_by_type(units, base_name)

    elif ext == ".docx":
        raw = _load_docx(file_path)
        units = _semantic_segment(raw)
        all_chunks = _chunk_by_type(units, base_name)

    else:
        raise ValueError(f"Formato no soportado: {ext}")

    print(f"[LOADER] {len(all_chunks)} unidades documentales generadas")
    if all_chunks:
        print("[LOADER] Ejemplo:")
        print(all_chunks[0]["type"], "→", all_chunks[0]["content"][:300])

    return all_chunks
