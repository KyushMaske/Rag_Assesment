import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import filter_complex_metadata
from src.config import settings


def get_vectorstore(documents=None):
    embedding_func = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    if documents:
        cleaned_docs = filter_complex_metadata(documents)
        vectorstore = FAISS.from_documents(
            documents=cleaned_docs, embedding=embedding_func
        )
        vectorstore.save_local(settings.FAISS_INDEX_DIR)
        return vectorstore

    if os.path.exists(settings.FAISS_INDEX_DIR):
        return FAISS.load_local(
            settings.FAISS_INDEX_DIR,
            embedding_func,
            allow_dangerous_deserialization=True,
        )

    return None
