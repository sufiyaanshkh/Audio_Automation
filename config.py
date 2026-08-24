import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
WORK_DIR = BASE_DIR / "work"

for directory in (UPLOAD_DIR, OUTPUT_DIR, WORK_DIR):
    directory.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN") or None

SRAVAANI_MODEL = os.getenv("SRAVAANI_MODEL", "ARTPARK-IISc/SraVaani-1.0")
LID_MODEL = os.getenv("LID_MODEL", "ARTPARK-IISc/Vaani-LID_v0")
TIMESTAMP_ASR_MODEL = os.getenv("TIMESTAMP_ASR_MODEL", "openai/whisper-small")
INDIC_EN_MODEL = os.getenv("INDIC_EN_MODEL", "ai4bharat/indictrans2-indic-en-1B")
EN_INDIC_MODEL = os.getenv("EN_INDIC_MODEL", "ai4bharat/indictrans2-en-indic-1B")
INDIC_INDIC_MODEL = os.getenv("INDIC_INDIC_MODEL", "ai4bharat/indictrans2-indic-indic-1B")

CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "15"))
LID_MIN_CONFIDENCE = float(os.getenv("LID_MIN_CONFIDENCE", "0.45"))
WHISPER_ENABLED = os.getenv("WHISPER_ENABLED", "true").lower() == "true"

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "mov", "webm", "ogg", "flac", "aac"}
