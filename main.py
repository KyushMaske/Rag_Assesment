import os
from src.config import settings, logger
from src.parser import extract_elements
from src.vectorstore import get_vectorstore
from src.engine import get_rag_chain


def main():
    if not os.path.exists(settings.FAISS_INDEX_DIR):
        logger.info("FAISS Index not found. Starting PDF Ingestion...")
        try:
            docs = extract_elements(settings.PDF_PATH)
            vectorstore = get_vectorstore(docs)
        except Exception as e:
            logger.error(f"Critical error during ingestion: {e}")
            return
    else:
        logger.info("Loading existing FAISS Index...")
        vectorstore = get_vectorstore()

    rag_chain = get_rag_chain(vectorstore)

    chat_history = []

    print("\n" + "=" * 50)
    print("RAG by SmartDataSolutionsLLC")
    print("=" * 50)
    print("Type 'exit' to quit. Follow-up questions are supported!")

    while True:
        query = input("\nUser: ").strip()

        if not query:
            continue
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        try:
            answer = rag_chain.invoke({"input": query, "chat_history": chat_history})
            print(f"\nAI: {answer}")

            chat_history.append(("human", query))
            chat_history.append(("assistant", answer))

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            print("\nAI: I'm sorry, I encountered an internal error. Please try again.")


if __name__ == "__main__":
    main()
