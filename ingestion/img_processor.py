# ingestion/image_processor.py
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
    Extrae imágenes de PDFs y genera descripciones textuales generales mediante un modelo visual-lingüístico.
    No depende de pies de figura ni términos específicos.
    Devuelve estructuras enriquecidas con coordenadas y descripciones listas para integrar con el texto.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        min_width: int = 200,
        min_height: int = 200,
        cache_path: str = "data/image_cache.json",
    ):
        DEEPINFRA_TOKEN = os.getenv("DEEPINFRA_TOKEN")
        if not DEEPINFRA_TOKEN:
            raise ValueError("No se encontró la variable DEEPINFRA_TOKEN en el .env")

        self.client = OpenAI(
            api_key=DEEPINFRA_TOKEN,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = model
        self.min_width = min_width
        self.min_height = min_height
        self.cache_path = cache_path
        self.cache = self._load_cache()

    # ------------------------
    # Cache helpers
    # ------------------------
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _hash_image(self, image_bytes: bytes) -> str:
        """Devuelve un hash único para identificar imágenes repetidas."""
        return hashlib.md5(image_bytes).hexdigest()

    # ------------------------
    # Descripción de imagen (agnóstica)
    # ------------------------
    def _describe_image(self, image_bytes: bytes) -> str:
        """
        Genera una descripción textual general de la imagen.
        No depende del dominio ni del tipo de contenido.
        """
        image_hash = self._hash_image(image_bytes)

        # Caché local
        if image_hash in self.cache:
            print(f"[IMG] Descripción recuperada de caché ({image_hash[:8]}...)")
            return self.cache[image_hash]

        try:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            prompt = """
            Describe detalladamente esta imagen de manera objetiva, sin hacer suposiciones.
            Incluye lo siguiente en un único bloque de texto:
            - Qué se observa en términos generales.
            - Qué elementos, estructuras o dispositivos son visibles.
            - Cómo se distribuyen espacialmente los componentes.
            - Cualquier texto o indicador visible (si lo hubiera).
            - Sin numeraciones ni listas, solo descripción continua en lenguaje técnico y neutral.
            """

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
            print(f"[IMG] Nueva descripción generada ({image_hash[:8]}...)")
            return description

        except Exception as e:
            print(f"[IMG] Error describiendo imagen: {e}")
            return ""

    # ------------------------
    # Procesamiento completo PDF
    # ------------------------
    def process_pdf_images(self, file_path: str) -> list[dict]:
        """
        Extrae imágenes del PDF con coordenadas, dimensiones y descripciones.
        Devuelve lista de dicts con:
        {
            page, x0, y0, x1, y1, description, hash, index
        }
        """
        print(f"[IMG] Analizando imágenes en {file_path} ...")
        results = []
        processed_hashes = set()

        try:
            with fitz.open(file_path) as pdf:
                for page_index, page in enumerate(pdf):
                    images = page.get_images(full=True)
                    if not images:
                        continue

                    for img_index, img in enumerate(images):
                        xref = img[0]
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image["image"]
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)

                        if width < self.min_width or height < self.min_height:
                            continue

                        image_hash = self._hash_image(image_bytes)
                        if image_hash in processed_hashes:
                            continue
                        processed_hashes.add(image_hash)

                        # Convertir a PNG (para consistencia)
                        try:
                            pil_img = Image.open(BytesIO(image_bytes))
                            buffer = BytesIO()
                            pil_img.save(buffer, format="PNG")
                            image_bytes = buffer.getvalue()
                        except Exception:
                            pass

                        # Obtener coordenadas visuales
                        rects = page.get_image_rects(xref)
                        if not rects:
                            continue
                        rect = rects[0]

                        # Generar descripción textual
                        description = self._describe_image(image_bytes)
                        if not description:
                            continue

                        image_info = {
                            "page": page_index + 1,
                            "x0": rect.x0,
                            "y0": rect.y0,
                            "x1": rect.x1,
                            "y1": rect.y1,
                            "hash": image_hash,
                            "index": img_index + 1,
                            "description": description,
                        }

                        results.append(image_info)

                        print(
                            f"[IMG] Imagen descrita (pág. {page_index + 1}): "
                            f"{description[:120].replace(chr(10),' ')}"
                        )

            print(f"[IMG] Total de imágenes descritas: {len(results)}")
            return results

        except Exception as e:
            print(f"[IMG] Error procesando PDF {file_path}: {e}")
            return []
