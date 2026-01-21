"""
Vector store management module for document embeddings.

This module handles creation, storage, and retrieval of document embeddings
using FAISS vector database.
"""

import os
from typing import List, Optional
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from src.config import settings, logger


def get_vectorstore(documents: Optional[List[Document]] = None) -> Optional[FAISS]:

    try:
        embedding_func = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f" Using embedding model: {settings.EMBEDDING_MODEL}")

        if documents:
            logger.info(f"Creating vector store from {len(documents)} documents.")

            cleaned_docs = filter_complex_metadata(documents)

            vectorstore = FAISS.from_documents(
                documents=cleaned_docs, embedding=embedding_func
            )

            os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
            vectorstore.save_local(settings.FAISS_INDEX_DIR)
            logger.info(f"Vector store saved to: {settings.FAISS_INDEX_DIR}")

            return vectorstore

        index_path = Path(settings.FAISS_INDEX_DIR)
        if index_path.exists():
            logger.info(
                f"Loading existing vector store from: {settings.FAISS_INDEX_DIR}"
            )

            vectorstore = FAISS.load_local(
                settings.FAISS_INDEX_DIR,
                embedding_func,
                allow_dangerous_deserialization=True,
            )

            index_size = vectorstore.index.ntotal
            logger.info(f"Loaded vector store with {index_size} vectors")

            return vectorstore

        logger.warning("No existing vector store found and no documents provided")
        return None

    except Exception as e:
        logger.error(f"Error with vector store: {str(e)}")
        raise


def get_retriever(
    vectorstore: FAISS,
    search_type: str = "similarity",
    k: int = 5,
    score_threshold: Optional[float] = None,
):

    search_kwargs = {"k": k}

    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    logger.info(f"Creating retriever: type={search_type}, k={k}")

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )

    return retriever


def delete_vectorstore() -> bool:
    """
    Deletes the existing vector store from disk.

    Returns:
        bool: True if deletion successful, False otherwise
    """
    import shutil

    index_path = Path(settings.FAISS_INDEX_DIR)

    if index_path.exists():
        try:
            shutil.rmtree(index_path)
            logger.info(f"Deleted vector store at: {settings.FAISS_INDEX_DIR}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")
            return False
    else:
        logger.warning("No vector store found to delete")
        return False


def get_vectorstore_info(vectorstore: FAISS) -> dict:

    try:
        info = {
            "total_vectors": vectorstore.index.ntotal,
            "embedding_dimension": vectorstore.index.d,
            "index_type": type(vectorstore.index).__name__,
        }
        return info
    except Exception as e:
        logger.error(f"Error getting vector store info: {e}")
        return {}
