# ---------- Dockerfile (entorno de desarrollo RAG) ----------
FROM python:3.10-slim AS base

# Variables de entorno básicas
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements y los instala
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Crea carpeta de datos (persistente)
RUN mkdir -p /app/data/chroma

# Copia el resto del proyecto (opcional: puedes omitir para montar con -v)
COPY . .

# Puerto de Streamlit
EXPOSE 8501

# Comando por defecto: shell interactivo
CMD ["/bin/bash"]
