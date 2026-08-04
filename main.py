from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from paddleocr import PaddleOCR
import numpy as np
import cv2
import json
import base64
import os
from openai import OpenAI

app = FastAPI(title="Servicio OCR de Facturas")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando modelo PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, lang='es')

class InvoiceRequest(BaseModel):
    imageBase64: str
    prompt: str = None
    apiKey: str = None

def organizar_texto_ocr(resultados_ocr):
    lineas = []
    for res in resultados_ocr:
        if not res: continue
        for caja in res:
            coordenadas, (texto, confianza) = caja
            promedio_y = sum([pt[1] for pt in coordenadas]) / 4
            lineas.append({'texto': texto, 'y': promedio_y})
    
    lineas.sort(key=lambda x: x['y'])
    texto_completo = "\n".join([item['texto'] for item in lineas])
    return texto_completo

@app.post("/api/scan-invoice")
async def scan_invoice(req: InvoiceRequest):
    try:
        # A. Leer la imagen desde el Base64 que envía Next.js
        image_data = base64.b64decode(req.imageBase64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagen no válida")

        # B. Extraer texto con PaddleOCR
        result = ocr.ocr(img, cls=True)
        texto_crudo = organizar_texto_ocr(result)
        
        # C. Configurar el cliente de DeepSeek
        api_key_to_use = req.apiKey if req.apiKey else os.getenv("DEEPSEEK_API_KEY")
        deepseek_client = OpenAI(
            api_key=api_key_to_use,
            base_url="https://api.deepseek.com"
        )
        
        # D. Configurar el prompt para maximizar Context Caching
        instrucciones = req.prompt if req.prompt else """
        Eres experto contable en Venezuela. Extrae datos de la factura a partir de este texto extraído por OCR.
        Reglas:
        1. Identifica si es de productos (type=1) o servicios (type=2) y aplícalo a todo.
        2. Agrupa montos por alícuota IVA (16%, 8%, 32%) o Exento (0%). Usa BI como priceU/total para gravables y Exento para exentos. NO listes productos individuales.
        3. El RIF (rifSupplier) es EXCLUSIVAMENTE el del comercio emisor (proveedor). Generalmente se encuentra en la parte superior (puede estar justo arriba o debajo del nombre del comercio). NO lo confundas con el RIF del cliente/comprador, el cual suele estar más abajo junto a etiquetas como "Cliente", "Razón Social" o "RIF/C.I.".
        4. typeDocument: 1(Factura), 2(N.Crédito), 3(N.Débito).
        5. dateShop formato YYYY-MM-DD.
        6. numCont es el Número de Control (ej. 00-00123). Si no hay un número de control claro impreso, utiliza el serial de la máquina fiscal (suele aparecer al final del ticket, ej. Z1F0021845).

        Debes devolver un JSON estrictamente con esta estructura (este es un ejemplo):
        {{
            "rifSupplier": "J12345678",
            "numDoc": "0001234",
            "numCont": "00-1234",
            "typeDocument": 1,
            "dateShop": "2023-10-25",
            "products": [
                {{
                    "status": 1,
                    "type": 1,
                    "name": "Monto Gravable 16%",
                    "cant": 1,
                    "priceU": 100.0,
                    "iva": 16.0,
                    "isExento": false,
                    "total": 100.0
                }}
            ]
        }}

        """

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that strictly outputs JSON.\n\n{instrucciones}"},
                {"role": "user", "content": f"Texto OCR a procesar:\n{texto_crudo}"}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        
        # E. Extraer y devolver el JSON
        json_content = response.choices[0].message.content
        datos_extraidos = json.loads(json_content)
        
        return {
            "success": True,
            "data": datos_extraidos
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
