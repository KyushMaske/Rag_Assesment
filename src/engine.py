from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.config import settings, logger, load_prompts

def format_docs(docs):
    formatted = []
    for doc in docs:
        dtype = doc.metadata.get("element_type", "Text")
        page = doc.metadata.get("page_number", "?")
        content = f"[{dtype} - Page {page}]:\n{doc.page_content}"
        formatted.append(content)
    return "\n\n---\n\n".join(formatted)

def get_rag_chain(vectorstore):
    logger.info("Initializing Conversational  Engine...")
    
    prompts = load_prompts()
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY, 
        model_name=settings.GROQ_MODEL, 
        temperature=0.1
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    contextualize_q_system_prompt = prompts["rag_system"]["contextualize_instruction"]
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    qa_system_prompt = prompts["rag_system"]["system_prompt"]
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def get_search_query(input_dict):
        if input_dict.get("chat_history") and len(input_dict["chat_history"]) > 0:
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
    
    logger.info("Conversational Chain Ready.")
    return rag_chain