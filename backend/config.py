
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

APP_NAME = os.getenv("APP_NAME", "NEXUS AI")
APP_ENV = os.getenv("APP_ENV", "development")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

DATA_DIR = ROOT_DIR / "data"
DATABASE_DIR = ROOT_DIR / "database"

DATA_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)
