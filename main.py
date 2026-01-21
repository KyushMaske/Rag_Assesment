'''
Main application file for the RAG pipeline.

This script initializes the RAG chain, handles PDF ingestion,
and manages user interactions via a command-line interface.
'''

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
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            
            answer = result["answer"]
            sources = result["source_documents"]
            print(f"\nAI: {answer}")
            
            if sources:
                print(f"\n[Sources used: {len(sources)} chunks]")
                for i, doc in enumerate(sources[:3]): # Show top 3 sources
                    pg = doc.metadata.get("page_number", "?")
                    dtype = doc.metadata.get("element_type", "Text")
                    page_content = doc.page_content.replace("\n", " ")
                    print(f" - Content: {page_content[:100]}...") 
                    print(f" - Source {i+1}: Page {pg}")

            chat_history.append(("human", query))
            chat_history.append(("assistant", answer))

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            print("\nAI: I'm sorry, I encountered an internal error. Please try again.")


if __name__ == "__main__":
    main()
