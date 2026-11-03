"""Descarga los modelos de PaddleOCR (lang='es', con clasificador de ángulo) al construir la imagen Docker.

Se ejecuta desde el Dockerfile (CDC-F-026). Antes los modelos se bajaban en cada arranque del contenedor.
Inicializar PaddleOCR durante el build se cae con segfault (código 139), así que se descargan los mismos
archivos a las mismas carpetas que usa PaddleOCR 2.8.1; al arrancar los encuentra y no descarga nada.
"""
import os
import tarfile
import tempfile
import urllib.request

BASE = os.path.expanduser("~/.paddleocr/whl")
MODELS = [
    ("https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
     os.path.join(BASE, "det", "en", "en_PP-OCRv3_det_infer")),
    ("https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
     os.path.join(BASE, "rec", "latin", "latin_PP-OCRv3_rec_infer")),
    ("https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
     os.path.join(BASE, "cls", "ch_ppocr_mobile_v2.0_cls_infer")),
]
FILES = ("inference.pdiparams", "inference.pdiparams.info", "inference.pdmodel")

for url, target in MODELS:
    os.makedirs(target, exist_ok=True)
    if all(os.path.exists(os.path.join(target, f)) for f in FILES[::2]):
        print("ya existe", target)
        continue
    print("descargando", url)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = os.path.join(tmp, "model.tar")
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                name = os.path.basename(member.name)
                if name in FILES and member.isfile():
                    with tar.extractfile(member) as src, open(os.path.join(target, name), "wb") as dst:
                        dst.write(src.read())
    missing = [f for f in FILES[::2] if not os.path.exists(os.path.join(target, f))]
    if missing:
        raise SystemExit(f"Faltan archivos del modelo en {target}: {missing}")
print("modelos listos")
