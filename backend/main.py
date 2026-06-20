import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# ==============================================================================
# CONFIGURACIÓN DE LOGGING
# ==============================================================================
# Configuramos el logging estándar de Python para poder visualizar en consola
# el inicio/fin del procesamiento y tener una buena trazabilidad de rendimiento.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# VARIABLES GLOBALES Y CONFIGURACIÓN DEL MODELO DE IA
# ==============================================================================
# Utilizamos "base" (o "small") para pruebas rápidas en local como se solicitó.
MODEL_SIZE = "base"

# En Apple Silicon M2 usamos CPU ya que CTranslate2 (el motor subyacente de 
# faster-whisper) está muy optimizado para ARM64. "compute_type=default" permite
# que el motor decida el tipo de dato óptimo (generalmente float32/int8).
DEVICE = "cpu"
COMPUTE_TYPE = "default"

# Variable global para mantener el modelo cargado en memoria y evitar
# tener que instanciarlo en cada petición HTTP, lo cual sería muy lento.
model = None

# ==============================================================================
# LIFESPAN (Gestión del ciclo de vida de FastAPI)
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.
    Nos permite cargar el modelo en memoria justo al arrancar el servidor
    (cold start), dejándolo listo para servir las peticiones de inmediato.
    """
    global model
    logger.info(f"Iniciando servidor... Cargando modelo '{MODEL_SIZE}' de faster-whisper en {DEVICE}.")
    start_time = time.time()
    
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        load_time = time.time() - start_time
        logger.info(f"Modelo cargado exitosamente en {load_time:.2f} segundos.")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise e
        
    yield # En este punto el servidor está arriba y aceptando peticiones
    
    # Limpieza cuando el servidor se apaga
    logger.info("Apagando servidor... Liberando recursos.")
    model = None

# ==============================================================================
# INSTANCIA PRINCIPAL DE FASTAPI
# ==============================================================================
app = FastAPI(
    title="Submagic Clone Backend API",
    description="Fase 1: API de transcripción de video a nivel de palabra usando Faster-Whisper",
    version="1.0.0",
    lifespan=lifespan
)

# ==============================================================================
# CONFIGURACIÓN DE CORS
# ==============================================================================
# Habilitamos CORS (Cross-Origin Resource Sharing) para que el frontend
# (corriendo en localhost:3000) no sea bloqueado por el navegador al
# intentar comunicarse con esta API (corriendo en localhost:8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Permite cualquier header
)

# ==============================================================================
# SERVICIOS (Lógica de Negocio Separada)
# ==============================================================================
def process_video_transcription(file_path: str) -> dict:
    """
    Recibe la ruta de un archivo de video, extrae el audio internamente 
    (gracias a la integración nativa de faster-whisper con FFmpeg) y 
    devuelve la transcripción estructurada.
    """
    global model
    if model is None:
        raise RuntimeError("El modelo de transcripción no se encuentra en memoria.")

    logger.info("Iniciando transcripción con faster-whisper...")
    start_time = time.time()

    # Transcribimos activando 'word_timestamps=True' según el requerimiento.
    # El generador comenzará a procesar el audio extraído con FFmpeg.
    segments_generator, info = model.transcribe(
        file_path, 
        word_timestamps=True,
        vad_filter=True  # Filtro de actividad de voz (VAD) para ignorar silencios y optimizar
    )
    
    logger.info(f"Idioma detectado: '{info.language}' (Probabilidad: {info.language_probability:.2f})")

    full_text = ""
    words_data = []

    # Iteramos sobre los segmentos generados. La inferencia real ocurre a medida
    # que iteramos sobre este generador.
    for segment in segments_generator:
        full_text += segment.text + " "
        
        # Iteramos sobre las palabras de cada segmento para extraer sus marcas de tiempo
        for word in segment.words:
            words_data.append({
                "word": word.word.strip(),
                "start": round(word.start, 3),
                "end": round(word.end, 3)
            })

    inference_time = time.time() - start_time
    logger.info(f"Transcripción completada exitosamente en {inference_time:.2f} segundos.")

    return {
        "text": full_text.strip(),
        "words": words_data,
        "language": info.language,
        "processing_time_seconds": round(inference_time, 2)
    }

# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.post("/api/v1/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    """
    Endpoint POST principal.
    Acepta un archivo de video temporal, ejecuta la inferencia y devuelve la
    transcripción estructurada a nivel de palabra.
    """
    logger.info(f"Petición POST a /api/v1/transcribe recibida. Archivo: {file.filename}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo.")

    # 1. Guardar el archivo de video temporalmente
    try:
        # mkstemp garantiza un nombre único y seguro en el sistema de archivos
        fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
        logger.info(f"Guardando video temporalmente en: {temp_path}")
        
        with os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
    except Exception as e:
        logger.error(f"Error al guardar el archivo en disco: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la subida del video.")

    # 2. Inferencia de IA
    try:
        result = process_video_transcription(temp_path)
    except Exception as e:
        logger.error(f"Error durante el procesamiento del modelo: {e}")
        # En caso de error, aseguramos la limpieza del archivo
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail="Error interno en la inferencia del modelo.")

    # 3. Limpieza de almacenamiento (No saturar disco local)
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Limpieza exitosa: Archivo {temp_path} eliminado.")
    except Exception as e:
        logger.warning(f"Advertencia: No se pudo eliminar el archivo temporal {temp_path}. Error: {e}")

    # 4. Retorno JSON Formateado para consumo en el Frontend
    return JSONResponse(content=result)
