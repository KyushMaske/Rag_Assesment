"""
Configuration module.

This module handles all environment variables and settings required
for the RAG pipeline including API keys, model configurations, and paths.
"""

from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
import logging
import tomllib
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    """

    GROQ_API_KEY: str
    GROQ_MODEL: str
    EMBEDDING_MODEL: str
    FAISS_INDEX_DIR: str
    PDF_PATH: str

    POPPLER_PATH: Optional[str] = None
    TESSERACT_PATH: Optional[str] = None
    PROMPTS_FILE: str = "prompts.toml"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def validate_paths(self) -> bool:
        paths_to_check = {
            "POPPLER_PATH": self.POPPLER_PATH,
            "TESSERACT_PATH": self.TESSERACT_PATH,
            "PROMPTS_FILE": self.PROMPTS_FILE,
        }

        all_valid = True
        for name, path in paths_to_check.items():
            if not Path(path).exists():
                logger.warning(f"{name} not found at: {path}")
                all_valid = False

        return all_valid


settings = Settings()


def load_prompts() -> Dict[str, Any]:
    prompts_path = Path(settings.PROMPTS_FILE)

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {settings.PROMPTS_FILE}")

    try:
        with open(prompts_path, "rb") as f:
            prompts = tomllib.load(f)
        logger.info(f"Loaded prompts from {settings.PROMPTS_FILE}")
        return prompts
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error parsing TOML file: {e}")
        raise


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RAG-Assignment")

if not settings.validate_paths():
    logger.warning(" Some paths are invalid. Please check your .env file.")
