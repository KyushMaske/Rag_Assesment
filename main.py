import os
from src.config import settings, logger
from src.parser import extract_elements
from src.vectorstore import get_vectorstore
from src.engine import get_rag_chain


def main():
    if not os.path.exists(settings.FAISS_INDEX_DIR):
        logger.info(" FAISS Index not found. Processing PDF...")
        docs = extract_elements(settings.PDF_PATH)
        vectorstore = get_vectorstore(docs)
    else:
        logger.info("Loading existing FAISS Index...")
        vectorstore = get_vectorstore()

    if not vectorstore:
        logger.error("Failed to initialize vectorstore.")
        return

    rag_chain = get_rag_chain(vectorstore)

    print("\n--- Apple 10-Q Analysis System Ready ---")
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        try:
            response = rag_chain.invoke(query)
            print(f"\nAnswer: {response}")
        except Exception as e:
            logger.error(f"Error during query: {e}")
            print("Please check  API key or logs.")


if __name__ == "__main__":
    main()
