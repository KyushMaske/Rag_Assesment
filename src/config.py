from pydantic_settings import BaseSettings
import logging
import tomllib

class Settings(BaseSettings):
    GROQ_API_KEY: str
    GROQ_MODEL: str
    EMBEDDING_MODEL: str 
    FAISS_INDEX_DIR: str 
    PDF_PATH: str 

    POPPLER_PATH: str
    TESSERACT_PATH: str
    PROMPTS_FILE: str = "prompts.toml"

    class Config:
        env_file = ".env"


settings = Settings()

def load_prompts():
    with open(settings.PROMPTS_FILE, "rb") as f:
        return tomllib.load(f)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AppleRAG")
