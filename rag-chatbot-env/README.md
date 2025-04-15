# Dr.X RAG Chatbot

A FastAPI-based chatbot that processes PDF, DOCX, and XLSX files, answers queries in English and Arabic, and summarizes responses. Uses FAISS for retrieval, Ollama (`llama3.2`) for answers, and Hugging Face for translation and summarization.

## Features
- Upload and process PDF, DOCX, XLSX files.
- Answer queries in English/Arabic with translation.
- Retrieve document chunks using FAISS.
- Summarize context/answers with ROUGE scores.
- Extract entities for English queries (spaCy).
- Redact PII (Presidio).
- Keep last 3 interactions (30-min timeout).

## Requirements
- Python 3.10+
- Ollama with `llama3.2`
- Windows/Linux/macOS

## Setup
1. **Clone Repo**:
   ```bash
   git clone <repo-url>
   cd Dr.X
   ```

2. **Virtual Environment**:
   ```bash
   python -m venv rag-chatbot-env
   .\rag-chatbot-env\Scripts\activate  # Windows
   ```

3. **Install Packages**:
   ```bash
   pip install fastapi uvicorn pdfplumber python-docx pandas langchain-community sentence-transformers faiss-cpu presidio-analyzer presidio-anonymizer spacy transformers torch langdetect rouge-score ollama
   python -m spacy download en_core_web_sm
   ```

4. **Set Up Ollama**:
   ```bash
   ollama pull llama3.2
   ollama serve
   ```

## Usage
1. **Run Server**:
   ```bash
   cd D:\Onedrive\OneDrive - IGO Solutions Ltd\Documents\AIGO Base\Code Base\Dr.X
   .\rag-chatbot-env\Scripts\activate
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Ingest File**:
   ```bash
   curl -X POST http://localhost:8000/ingest -F "file=@sample.pdf"
   ```

3. **Query**:
   - English:
     ```bash
     curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"What are new cancer treatments?\", \"top_k\": 5, \"target_language\": \"en\"}"
     ```
   - Arabic:
     ```bash
     curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"ما هي العلاجات الجديدة للسرطان؟\", \"top_k\": 5, \"target_language\": \"ar\"}"
     ```

## Structure
```
Dr.X/
├── data/
│   ├── faiss_index.bin
│   ├── faiss_metadata.pkl
├── rag-chatbot-env/
├── main.py
├── README.md
```

## Troubleshooting
- **Arabic Translation Error**:
   ```bash
   pip install transformers sentencepiece
   rmdir /S /Q %USERPROFILE%\.cache\huggingface\hub
   ```
- **Ollama**:
   ```bash
   ollama list
   ollama pull llama3.2
   ```
- Check logs:
   ```bash
   uvicorn main:app --log-level debug
   ```

