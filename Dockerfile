FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema requeridas por PaddleOCR y OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Descargar los modelos de PaddleOCR al construir la imagen (CDC-F-026). Antes se bajaban en cada arranque
# del contenedor: cada reinicio tardaba más y fallaba si no había conexión a los servidores de Paddle.
COPY download_models.py .
RUN python download_models.py

# Copiar el resto del código
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
