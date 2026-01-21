# RAG Pipeline

A high-performance Retrieval-Augmented Generation (RAG) system designed to process complex PDF documents—including text, tables, and figures—using **LangChain**, **FAISS**, and **Groq**. 


## Features

- **Advanced PDF Parsing**: Uses the `Unstructured` library with a `hi_res` strategy to extract tables (as HTML) and narrative text accurately.
- **OCR Integration**: Built-in support for **Tesseract** and **Poppler** to handle scanned documents.
- **Contextual Querying**: Implements a two-step LCEL (LangChain Expression Language) chain that rewrites follow-up questions to be standalone queries.
- **Fast Inference**: Powered by **Groq ** for near-instantaneous LLM responses.
- **Dual Interface**:
    - **Streamlit Web UI**: For a modern, interactive chat experience with source citations.
    - **CLI Interface**: For lightweight, terminal-based interactions.
- **Persistent Vector Store**: Uses **FAISS** to store and reload document embeddings locally.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KyushMaske/Rag_Assesment.git
   cd rag-Assesment
   ```

2. **Install Python Dependencies**:
   ```bash
   # Create a virtual environment
   uv venv

   # Activate the environment

   # On Windows:
   .venv\Scripts\activate

   # On macOS/Linux:
   source .venv/bin/activate

   # Install dependencies
   uv pip install -r requirements.txt
   ```


3. **Install System Dependencies**:
    - **Ubuntu/Debian**:
      ```bash
      sudo apt-get install poppler-utils tesseract-ocr
      ```
    - **Windows**: 
      Download and install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) and [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Add their `bin` folders to your System PATH or specify them in the `.env` file.  

---

## Configuration

Create a `.env` file in the root directory and fill in your details:

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here

# Model Settings
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Paths
FAISS_INDEX_DIR=faiss_index
PDF_PATH=data/<PDF_PATH>
PROMPTS_FILE=prompts.toml

# External Tools (Optional: provide paths if not in System PATH)
POPPLER_PATH=C:/Program Files/poppler-25.12.0/Library/bin
TESSERACT_PATH=C:/Program Files/Tesseract-OCR
```

You can download Faiss index(named as faiss_index) from [here](https://drive.google.com/drive/folders/1pe0cbd0-yAXkiPSRt20D74GJbvK_uDGL?usp=sharing)
---

## Running the Application

### 1. Streamlit Web Interface (Recommended)
This provides a chat UI and allows you to upload new PDFs dynamically.
```bash
uv run streamlit run app.py
```

### 2. CLI Interface
Use this for quick testing or terminal-based interaction.
```bash
uv run main.py
```


