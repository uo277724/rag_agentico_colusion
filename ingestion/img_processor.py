import fitz  # PyMuPDF
import base64
import hashlib
import json
import os
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class ImageProcessor:
    """
    Extrae imágenes de PDFs y genera descripciones visuales objetivas.
    Las imágenes se tratan como nodos documentales independientes,
    nunca como fuente primaria de información económica.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        min_width: int = 200,
        min_height: int = 200,
        cache_path: str = "data/image_cache.json",
    ):
        token = os.getenv("DEEPINFRA_TOKEN")
        if not token:
            raise ValueError("No se encontró la variable DEEPINFRA_TOKEN en el .env")

        self.client = OpenAI(
            api_key=token,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = model
        self.min_width = min_width
        self.min_height = min_height
        self.cache_path = cache_path
        self.cache = self._load_cache()

    # --------------------------------------------------
    # Cache helpers
    # --------------------------------------------------
    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _hash_image(self, image_bytes: bytes) -> str:
        return hashlib.md5(image_bytes).hexdigest()

    # --------------------------------------------------
    # Descripción visual neutra
    # --------------------------------------------------
    def _describe_image(self, image_bytes: bytes) -> str:
        image_hash = self._hash_image(image_bytes)

        if image_hash in self.cache:
            return self.cache[image_hash]

        try:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            prompt = (
                "Describe esta imagen de forma objetiva y técnica.\n"
                "No hagas inferencias ni interpretaciones.\n"
                "Limítate a lo visualmente observable, incluyendo texto visible si lo hay.\n"
                "Usa un único párrafo descriptivo."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            description = response.choices[0].message.content.strip()
            self.cache[image_hash] = description
            self._save_cache()
            return description

        except Exception as e:
            print(f"[IMG] Error describiendo imagen: {e}")
            return ""

    # --------------------------------------------------
    # Procesamiento completo de imágenes en PDF
    # --------------------------------------------------
    def process_pdf_images(self, file_path: str) -> list[dict]:
        """
        Devuelve imágenes como nodos documentales independientes:
        {
            type: "figure",
            page,
            bbox,
            description,
            hash,
            confidence
        }
        """
        print(f"[IMG] Analizando imágenes en {file_path} ...")
        results = []
        processed_hashes = set()

        try:
            with fitz.open(file_path) as pdf:
                for page_index, page in enumerate(pdf):
                    for img_index, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image.get("image", b"")

                        if not image_bytes:
                            continue

                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        if width < self.min_width or height < self.min_height:
                            continue

                        image_hash = self._hash_image(image_bytes)
                        if image_hash in processed_hashes:
                            continue
                        processed_hashes.add(image_hash)

                        # Normalizar formato
                        try:
                            pil_img = Image.open(BytesIO(image_bytes))
                            buffer = BytesIO()
                            pil_img.save(buffer, format="PNG")
                            image_bytes = buffer.getvalue()
                        except Exception:
                            pass

                        rects = page.get_image_rects(xref)
                        if not rects:
                            continue
                        rect = rects[0]

                        description = self._describe_image(image_bytes)
                        if not description:
                            continue

                        results.append({
                            "type": "figure",
                            "page": page_index + 1,
                            "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                            "description": description,
                            "hash": image_hash,
                            "confidence": 0.2  # Nunca fuente primaria
                        })

            print(f"[IMG] Total de imágenes procesadas: {len(results)}")
            return results

        except Exception as e:
            print(f"[IMG] Error procesando PDF {file_path}: {e}")
            return []
