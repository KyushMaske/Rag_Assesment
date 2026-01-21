'''
Main Streamlit application for the RAG pipeline.

This app allows users to upload PDFs, processes them,
and interact with the RAG chain via a chat interface.

'''

import streamlit as st
import os
from pathlib import Path
from src.config import settings, logger
from src.parser import extract_elements
from src.vectorstore import get_vectorstore
from src.engine import get_rag_chain

st.set_page_config(page_title="RAG :SmartDataSolutionsLLC", page_icon="📊", layout="wide")

st.title("RAG :  SmartDataSolutionsLLC")
st.subheader("Assignment submitted by Kyush Maskey")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
    else:
        st.info("No PDF uploaded yet.")
        
    
    if st.button("Rebuild Vector Store"):
        if uploaded_file:
            
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            file_path = data_dir / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Parsing PDF and generating embeddings..."):
                try:
                    docs = extract_elements(str(file_path))
                    vectorstore = get_vectorstore(docs)
                    st.session_state.rag_chain = get_rag_chain(vectorstore)
                    st.success("Vector store updated!")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please upload a PDF first.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if st.session_state.rag_chain is None:
    if os.path.exists(settings.FAISS_INDEX_DIR):
        with st.spinner("Loading existing index..."):
            vs = get_vectorstore()
            st.session_state.rag_chain = get_rag_chain(vs)
    else:
        st.info("Please upload a PDF and click 'Rebuild Vector Store' to start.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about the document..."):
    if st.session_state.rag_chain is None:
        st.error("RAG Chain is not initialized. Please process a document.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):

                chat_history = [
                    (m["role"], m["content"]) for m in st.session_state.messages[-10:]
                ]
                
                try:
                    result = st.session_state.rag_chain.invoke(
                        {"input": query, "chat_history": chat_history[:-1]} 
                    )
                    
                    answer = result["answer"]
                    sources = result["source_documents"]

                    st.markdown(answer)

                    if sources:
                        with st.expander(" View Sources & Context"):
                            for i, doc in enumerate(sources):
                                page = doc.metadata.get("page_number", "N/A")
                                dtype = doc.metadata.get("element_type", "Text")
                                st.markdown(f"**Source {i+1} | Page {page} | Type: {dtype}**")
                                st.caption(doc.page_content)
                                st.divider()
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    st.error(f"An error occurred: {str(e)}")