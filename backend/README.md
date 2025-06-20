# Inquiero Backend

This is the FastAPI backend for the Inquiero document-based Q&A system.

## Project Structure

```
backend/
├── config/                 # Configuration files
│   ├── __init__.py
│   └── settings.py
├── utils/                  # Core application modules
│   ├── __init__.py
│   ├── api.py             # FastAPI routes and endpoints
│   ├── models.py          # Pydantic models
│   ├── pdf_processor.py   # PDF processing and text extraction
│   ├── rag_engine.py      # Advanced RAG implementation
│   └── chat_manager.py    # Chat history management
├── data/                   # Data storage
│   └── pdfs/              # PDF files storage (single location)
├── chat_history/          # Chat history JSON files
├── run.py                 # Application entry point
├── requirements.txt       # Python dependencies
├── test_rag.py           # RAG testing script
└── .gitignore            # Git ignore rules
```

## PDF Storage Structure

**Single Source of Truth**: PDF files are stored in only one location: `data/pdfs/`

- **Upload**: When a PDF is uploaded, it's saved to `data/pdfs/` with a unique filename
- **Chat References**: Chat history files reference the original PDF paths instead of copying files
- **No Duplication**: This eliminates file duplication and saves storage space

### File Naming Convention
- PDFs are saved with format: `{uuid}_{original_filename}.pdf`
- Example: `a1b2c3d4_sample_document.pdf`

## Key Features

1. **PDF Processing**: Text extraction with OCR fallback
2. **Advanced RAG**: Hybrid dense/sparse retrieval with reranking
3. **Chat Management**: Persistent chat history with conversation memory
4. **Multi-language Support**: Automatic language detection
5. **Vector Storage**: FAISS-based similarity search

## API Endpoints

- `POST /upload-pdf/` - Upload and process PDF files
- `POST /chat/` - Send messages and get AI responses
- `GET /chats/` - Get all chat history
- `DELETE /chats/{chat_id}` - Delete specific chat
- `GET /rag/stats` - Get RAG engine statistics
- `POST /rag/clear-memory` - Clear RAG conversation memory

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the server:
   ```bash
   python run.py
   ```

3. The API will be available at `http://localhost:8000`

## Dependencies

- FastAPI - Web framework
- PyMuPDF - PDF processing
- LangChain - LLM integration
- FAISS - Vector similarity search
- Ollama - Local LLM (Mistral)
- Pydantic - Data validation

## Notes

- PDF files are automatically ignored by git to prevent large file commits
- Chat history is stored as JSON files for easy backup and migration
- The system uses local Ollama for LLM processing (requires Ollama to be running) 