"""
PDF parsing module for extracting structured content.

This module handles parsing of PDF documents including text, tables, and figures
using the Unstructured library with OCR capabilities.
"""

import os
import pytesseract
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from src.config import settings, logger


def setup_ocr_environment() -> None:

    tesseract_exe = Path(settings.TESSERACT_PATH) / "tesseract.exe"

    if tesseract_exe.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)
        logger.info(f"Tesseract configured at: {tesseract_exe}")
    else:
        logger.error(f" Tesseract not found at {tesseract_exe}. OCR will fail.")
        raise FileNotFoundError(f"Tesseract executable not found: {tesseract_exe}")

    poppler_path = Path(settings.POPPLER_PATH)
    if poppler_path.exists():
        os.environ["PATH"] += os.pathsep + str(poppler_path)
        logger.info(f"Poppler path configured: {poppler_path}")
    else:
        logger.error(f"Poppler path not found at {poppler_path}")
        raise FileNotFoundError(f"Poppler path not found: {poppler_path}")


try:
    setup_ocr_environment()
except FileNotFoundError as e:
    logger.warning(f"OCR setup incomplete: {e}")


def extract_elements(
    file_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    strategy: str = "hi_res",
    chunking_strategy: str = "by_title",
    max_characters: int = 2000,
) -> List[Document]:
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        logger.error(f"PDF file not found: {file_path}")
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    logger.info(f"Partitioning PDF: {file_path}")
    logger.info(f"Strategy: {strategy}, Max chars: {max_characters}")

    try:
        elements = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy=strategy,
            chunking_strategy=chunking_strategy,
            max_characters=max_characters,
        )

        logger.info(f"Extracted {len(elements)} elements from PDF")

        raw_docs = []
        element_counts = {"Text": 0, "Table": 0, "Figure": 0, "Other": 0}

        for el in elements:
            metadata = {
                "source": str(file_path_obj.name),
                "element_type": el.category,
                "page_number": (
                    el.metadata.page_number if el.metadata.page_number else 1
                ),
            }

            content = (
                el.metadata.text_as_html
                if el.category == "Table" and el.metadata.text_as_html
                else el.text
            )

            if content and content.strip():
                raw_docs.append(Document(page_content=content, metadata=metadata))
                element_counts[
                    el.category if el.category in element_counts else "Other"
                ] += 1

        logger.info(f"Element breakdown: {element_counts}")

        if not raw_docs:
            logger.error(f"No elements found in PDF: {file_path}")
            raise ValueError(f"No elements found in PDF: {file_path}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )

        final_docs = text_splitter.split_documents(raw_docs)
        logger.info(f" Created {len(final_docs)} chunks from {len(raw_docs)} elements")

        return final_docs

    except Exception as e:
        logger.error(f"Error during PDF parsing: {str(e)}")
        raise


def get_document_stats(documents: List[Document]) -> dict:

    if not documents:
        return {"total_docs": 0}

    element_types = {}
    total_chars = 0
    chunk_lengths = []
    pages = set()

    for doc in documents:

        elem_type = doc.metadata.get("element_type", "Unknown")
        element_types[elem_type] = element_types.get(elem_type, 0) + 1

        content_length = len(doc.page_content)
        total_chars += content_length
        chunk_lengths.append(content_length)

        pages.add(doc.metadata.get("page_number", 0))

    return {
        "total_docs": len(documents),
        "element_types": element_types,
        "total_characters": total_chars,
        "avg_chunk_length": total_chars // len(documents) if documents else 0,
        "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
        "total_pages": len(pages),
    }
