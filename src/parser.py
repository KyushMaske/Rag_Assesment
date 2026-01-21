import os
import pytesseract
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from src.config import settings, logger


tesseract_exe = os.path.join(settings.TESSERACT_PATH, "tesseract.exe")
if os.path.exists(tesseract_exe):
    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
else:
    logger.error(f"Tesseract not found at {tesseract_exe}. OCR will fail.")

if os.path.exists(settings.POPPLER_PATH):
    os.environ["PATH"] += os.pathsep + settings.POPPLER_PATH
else:
    logger.error(f"Poppler path not found at {settings.POPPLER_PATH}")


def extract_elements(file_path: str) -> List[Document]:
    """
    Parses a PDF into structured Documents.
    Captures Tables as HTML and Figures as text via OCR.
    """
    logger.info(f" Partitioning PDF: {file_path}")

    try:

        elements = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            chunking_strategy="by_title",
            max_characters=2000,
        )

        raw_docs = []
        for el in elements:

            metadata = {
                "source": file_path,
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

            if content.strip():
                raw_docs.append(Document(page_content=content, metadata=metadata))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", " "],
        )

        final_docs = text_splitter.split_documents(raw_docs)
        logger.info(f" Extracted {len(final_docs)} refined chunks from PDF.")
        return final_docs

    except Exception as e:
        logger.error(f"❌ Error during PDF parsing: {str(e)}")
        raise e
