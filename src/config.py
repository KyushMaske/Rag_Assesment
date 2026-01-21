from pydantic_settings import BaseSettings
import logging


class Settings(BaseSettings):
    GROQ_API_KEY: str
    GROQ_MODEL: str
    EMBEDDING_MODEL: str 
    FAISS_INDEX_DIR: str 
    PDF_PATH: str 

    POPPLER_PATH: str
    TESSERACT_PATH: str

    class Config:
        env_file = ".env"


settings = Settings()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AppleRAG")
