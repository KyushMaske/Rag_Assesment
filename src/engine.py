"""
RAG chain engine module.

This module implements the conversational RAG chain using LangChain Expression Language (LCEL),
handling context retrieval, query contextualization, and answer generation.
"""

from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.config import settings, logger, load_prompts


def validate_input(input_dict: Dict[str, Any]) -> bool:

    if not isinstance(input_dict, dict):
        logger.error("Input must be a dictionary")
        return False

    if "input" not in input_dict:
        logger.error("Input dictionary must contain 'input' key")
        return False

    if not input_dict["input"] or not input_dict["input"].strip():
        logger.error("Input cannot be empty")
        return False

    return True


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant context found."

    formatted = []
    for i, doc in enumerate(docs, 1):
        dtype = doc.metadata.get("element_type", "Text")
        page = doc.metadata.get("page_number", "?")
        source = doc.metadata.get("source", "Unknown")

        header = f"[Document {i} - {dtype} from {source}, Page {page}]"
        content = f"{header}\n{doc.page_content}"
        formatted.append(content)

    result = "\n\n" + "=" * 80 + "\n\n"
    result += "\n\n---\n\n".join(formatted)

    return result


def get_rag_chain(vectorstore, temperature: float = 0.1, k: int = 5):

    logger.info("Initializing Conversational RAG Chain...")

    try:
        prompts = load_prompts()

        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=temperature,
        )
        logger.info(f"LLM initialized: {settings.GROQ_MODEL} (temp={temperature})")

        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        logger.info(f"Retriever configured: k={k}")

        contextualize_q_system_prompt = prompts["rag_system"][
            "contextualize_instruction"
        ]
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

        qa_system_prompt = prompts["rag_system"]["system_prompt"]
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        def get_search_query(input_dict: Dict[str, Any]) -> str:

            if not validate_input(input_dict):
                raise ValueError("Invalid input for RAG chain")

            chat_history = input_dict.get("chat_history", [])

            if chat_history and len(chat_history) > 0:
                logger.info("Contextualizing question with chat history...")
                return contextualize_chain.invoke(input_dict)

            return input_dict["input"]

        rag_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(get_search_query) | retriever | format_docs
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Conversational RAG Chain ready")
        return rag_chain

    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        raise


def format_chat_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:

    formatted = []

    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            formatted.append({"role": msg["role"], "content": msg["content"]})

    return formatted
