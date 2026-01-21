from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import settings, logger


def format_docs(docs):
    formatted = []
    for doc in docs:
        dtype = doc.metadata.get("element_type", "Text")
        page = doc.metadata.get("page_number", "?")
        content = f"[{dtype} - Page {page}]:\n{doc.page_content}"
        formatted.append(content)
    return "\n\n---\n\n".join(formatted)


def get_rag_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL,
        temperature=0.1,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    system_prompt = (
        "You are an expert Financial Analyst. Use the provided context to answer the question.\n\n"
        "CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
