"""Servicio OCR de facturas (contrato C-04 del cerebro, mejora CDC-F-026).

Flujo: archivo (imagen o PDF) -> texto (capa de texto del PDF o PaddleOCR) -> DeepSeek -> JSON.

Variables de entorno (todas opcionales salvo DEEPSEEK_API_KEY):
  DEEPSEEK_API_KEY        clave de DeepSeek si el request no trae apiKey
  OCR_INTERNAL_TOKEN      si está definido, se exige "Authorization: Bearer <token>" (el mismo valor en el .env de Next)
  OCR_ALLOWED_ORIGINS     orígenes CORS separados por coma (vacío = sin CORS; Next llama desde el servidor)
  OCR_DET_LIMIT_SIDE_LEN  lado máximo para la detección de texto (por defecto 960; 1280 midió ~1,5 s más por foto)
  OCR_USE_ANGLE_CLS       "true"/"false": clasificador de texto girado (por defecto true)
  OCR_ENABLE_MKLDNN       "true"/"false": MKL-DNN (por defecto false; en la prueba local fue más lento)
  OCR_CPU_THREADS         hilos de CPU para PaddleOCR (por defecto todos los núcleos)
  PDF_DPI                 resolución al convertir un PDF escaneado a imagen (por defecto 200)
  LLM_TIMEOUT_SECONDS     tiempo máximo de espera a DeepSeek (por defecto 60)
"""
import base64
import hmac
import json
import logging
import os
import threading
import time
from typing import List, Optional

import cv2
import fitz  # PyMuPDF - para leer PDFs
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from paddleocr import PaddleOCR
from pydantic import BaseModel

# Logger propio: PaddleOCR reconfigura el logging y ocultaba los mensajes INFO (tiempos por etapa)
log = logging.getLogger("ocr-service")
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_handler)
log.propagate = False


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "si", "sí")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


INTERNAL_TOKEN = os.getenv("OCR_INTERNAL_TOKEN", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("OCR_ALLOWED_ORIGINS", "").split(",") if o.strip()]
PDF_DPI = env_int("PDF_DPI", 200)
LLM_TIMEOUT = env_int("LLM_TIMEOUT_SECONDS", 60)
MIN_PDF_TEXT_CHARS = 40  # menos que esto = PDF escaneado (imagen): se pasa por OCR

app = FastAPI(title="Servicio OCR de Facturas")

# Next llama a este servicio desde el servidor, así que no hace falta CORS. Solo se activa si se configura.
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

log.info("Cargando modelo PaddleOCR...")
ocr = PaddleOCR(
    use_angle_cls=env_bool("OCR_USE_ANGLE_CLS", True),
    lang="es",
    det_limit_side_len=env_int("OCR_DET_LIMIT_SIDE_LEN", 960),
    enable_mkldnn=env_bool("OCR_ENABLE_MKLDNN", False),
    cpu_threads=env_int("OCR_CPU_THREADS", os.cpu_count() or 4),
    show_log=False,
)
# PaddleOCR no es seguro entre hilos: se usa de a uno. La llamada a DeepSeek (la parte más lenta) sí corre en paralelo.
ocr_lock = threading.Lock()

# Un cliente de DeepSeek por clave, reutilizado entre peticiones (evita abrir una conexión nueva cada vez)
_llm_clients = {}
_llm_lock = threading.Lock()


def get_llm_client(api_key: str) -> OpenAI:
    with _llm_lock:
        client = _llm_clients.get(api_key)
        if client is None:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=LLM_TIMEOUT, max_retries=1)
            _llm_clients[api_key] = client
        return client


class InvoiceRequest(BaseModel):
    imageBase64: str
    fileType: str = "image"  # "image" o "pdf"
    prompt: Optional[str] = None
    apiKey: Optional[str] = None


DEFAULT_PROMPT = """
Eres experto contable en Venezuela. Extrae datos de la factura a partir de este texto extraído por OCR.
Devuelve un JSON con rifSupplier, numDoc, numCont, typeDocument (1 factura, 2 nota de crédito, 3 nota de débito),
dateShop (YYYY-MM-DD) y products agrupados por tipo (1 bien, 2 servicio) y alícuota de IVA.
"""


def organizar_texto_ocr(resultados_ocr) -> str:
    """Arma el texto renglón por renglón, de izquierda a derecha.

    Antes se ordenaba cada caja solo por su altura: la etiqueta ("BASE IMPONIBLE") y su monto quedaban en líneas
    distintas o mezclados con otra columna. Ahora las cajas que se solapan en vertical forman un mismo renglón.
    """
    cajas = []
    for res in resultados_ocr or []:
        if not res:
            continue
        for caja in res:
            coordenadas, (texto, _confianza) = caja
            ys = [pt[1] for pt in coordenadas]
            xs = [pt[0] for pt in coordenadas]
            y_min, y_max = min(ys), max(ys)
            cajas.append({"texto": texto, "x": min(xs), "yc": (y_min + y_max) / 2, "h": max(y_max - y_min, 1)})

    cajas.sort(key=lambda c: c["yc"])
    renglones: List[dict] = []
    for caja in cajas:
        actual = renglones[-1] if renglones else None
        # Mismo renglón si el centro vertical está dentro de media altura del renglón actual
        if actual and abs(caja["yc"] - actual["yc"]) <= 0.5 * min(caja["h"], actual["h"]):
            actual["cajas"].append(caja)
            n = len(actual["cajas"])
            actual["yc"] = (actual["yc"] * (n - 1) + caja["yc"]) / n
            actual["h"] = max(actual["h"], caja["h"])
        else:
            renglones.append({"yc": caja["yc"], "h": caja["h"], "cajas": [caja]})

    lineas = []
    for renglon in renglones:
        partes = sorted(renglon["cajas"], key=lambda c: c["x"])
        lineas.append("   ".join(p["texto"] for p in partes))
    return "\n".join(lineas)


def pdf_texto_digital(doc: "fitz.Document", max_paginas: int = 3) -> str:
    """Texto de un PDF digital (generado por un sistema). Vacío si el PDF es una imagen escaneada."""
    partes = []
    for i in range(min(doc.page_count, max_paginas)):
        partes.append(doc[i].get_text("text", sort=True))
    texto = "\n".join(partes).strip()
    return texto if len(texto) >= MIN_PDF_TEXT_CHARS else ""


def pdf_primera_pagina_a_imagen(doc: "fitz.Document") -> np.ndarray:
    """Convierte la primera página de un PDF en un array numpy (BGR) para PaddleOCR."""
    pagina = doc[0]
    mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
    pixmap = pagina.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def check_token(authorization: Optional[str]) -> None:
    if not INTERNAL_TOKEN:
        return  # sin token configurado se acepta como antes (compatibilidad con C-04 v1)
    expected = f"Bearer {INTERNAL_TOKEN}"
    if not authorization or not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="No autorizado")


@app.on_event("startup")
def warmup() -> None:
    """Primera pasada con una imagen vacía para que la primera factura real no pague la inicialización."""
    try:
        t0 = time.perf_counter()
        with ocr_lock:
            ocr.ocr(np.full((64, 256, 3), 255, dtype=np.uint8), cls=True)
        log.info("PaddleOCR listo en %.2fs", time.perf_counter() - t0)
    except Exception:  # el calentamiento nunca debe impedir que el servicio arranque
        log.exception("Falló el calentamiento de PaddleOCR")


@app.get("/health")
def health():
    return {"status": "ok"}


# Ruta síncrona a propósito: FastAPI la ejecuta en un hilo aparte, así una factura no bloquea a las demás
@app.post("/api/scan-invoice")
def scan_invoice(req: InvoiceRequest, authorization: Optional[str] = Header(default=None)):
    check_token(authorization)
    timings = {}
    t_total = time.perf_counter()

    # A. Decodificar el archivo
    t0 = time.perf_counter()
    try:
        file_data = base64.b64decode(req.imageBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo no válido o no se pudo decodificar")

    source = "ocr"
    texto_crudo = ""
    img = None
    try:
        if req.fileType == "pdf":
            doc = fitz.open(stream=file_data, filetype="pdf")
            try:
                if doc.page_count == 0:
                    raise HTTPException(status_code=400, detail="El PDF no contiene páginas")
                # PDF digital: el texto ya viene en el archivo, sin errores de OCR
                texto_crudo = pdf_texto_digital(doc)
                if texto_crudo:
                    source = "pdf-text"
                else:
                    img = pdf_primera_pagina_a_imagen(doc)
            finally:
                doc.close()
        else:
            img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Archivo no válido o no se pudo decodificar")
    except HTTPException:
        raise
    except Exception:
        log.exception("Error al leer el archivo")
        raise HTTPException(status_code=400, detail="Archivo no válido o no se pudo decodificar")
    timings["decode"] = round(time.perf_counter() - t0, 3)

    # B. OCR (solo si no había texto digital)
    if source == "ocr":
        t0 = time.perf_counter()
        try:
            with ocr_lock:
                timings["ocrWait"] = round(time.perf_counter() - t0, 3)
                t1 = time.perf_counter()
                result = ocr.ocr(img, cls=True)
                timings["ocr"] = round(time.perf_counter() - t1, 3)
        except Exception:
            log.exception("Error en PaddleOCR")
            raise HTTPException(status_code=500, detail="Error al leer el texto de la factura")
        texto_crudo = organizar_texto_ocr(result)

    if not texto_crudo.strip():
        raise HTTPException(status_code=422, detail="No se encontró texto en la factura")
    log.info("Texto listo source=%s timings=%s chars=%d", source, timings, len(texto_crudo))

    # C. Interpretar con DeepSeek. El prompt (instrucciones + ejemplos) va primero y fijo para aprovechar
    #    el caché de contexto de DeepSeek; el texto de la factura va al final.
    api_key = req.apiKey or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Servicio OCR sin clave de DeepSeek configurada")
    instrucciones = req.prompt or DEFAULT_PROMPT

    t0 = time.perf_counter()
    try:
        response = get_llm_client(api_key).chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that strictly outputs JSON.\n\n{instrucciones}"},
                {"role": "user", "content": f"Texto OCR a procesar:\n{texto_crudo}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1500,
        )
        datos_extraidos = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        log.exception("DeepSeek no devolvió JSON válido")
        raise HTTPException(status_code=502, detail="La IA no devolvió un JSON válido")
    except Exception:
        log.exception("Error al consultar DeepSeek")
        raise HTTPException(status_code=502, detail="Error al consultar el servicio de IA")
    timings["llm"] = round(time.perf_counter() - t0, 3)
    timings["total"] = round(time.perf_counter() - t_total, 3)

    usage = getattr(response, "usage", None)
    log.info("Factura procesada source=%s timings=%s tokens=%s", source, timings,
             getattr(usage, "total_tokens", None) if usage else None)

    return {
        "success": True,
        "data": datos_extraidos,
        # Campos agregados en CDC-F-026 (opcionales para el consumidor)
        "meta": {"source": source, "timings": timings, "ocrText": texto_crudo},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
