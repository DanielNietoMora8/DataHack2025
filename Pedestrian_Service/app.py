from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import os
import subprocess

app = FastAPI()

UPLOAD_DIR = "videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <h2>Sube un video para analizar</h2>
    <form action="/upload/" method="post" enctype="multipart/form-data">
        <input name="file" type="file">
        <input type="submit" value="Subir y procesar">
    </form>
    """

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print(f"📁 Video guardado en: {video_path}")

    # Lanza detection.py como proceso separado
    subprocess.Popen(["python", "detection.py", video_path])

    return {"message": "Video recibido. Procesando en segundo plano."}
