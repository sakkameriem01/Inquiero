# Inquiero Architecture

## Overview
Inquiero is a document-based Q&A system that processes PDF documents and enables interactive conversations about their content using AI.

## System Architecture

```
┌─────────────────┐     HTTP/WebSocket    ┌─────────────────┐
│                 │     Communication     │                 │
│    Frontend     │◄─────────────────────►│     Backend     │
│    (React)      │                       │    (FastAPI)     │
└─────────────────┘                       └─────────────────┘
        │                                          │
        │                                          │
        ▼                                          ▼
┌─────────────────┐                       ┌─────────────────┐
│  State          │                       │  PDF Processor  │
│  Management     │                       │  Service        │
└─────────────────┘                       └─────────────────┘
                                               │
                                               │
                                               ▼
┌─────────────────┐     ┌─────────────────┐    │    ┌─────────────────┐
│  Vector Store   │◄────┤  Text           │◄───┘    │  Chat History   │
│  (FAISS)        │     │  Embeddings     │         │  Service        │
└─────────────────┘     └─────────────────┘         └─────────────────┘
        │                      │                           │
        │                      │                           │
        ▼                      ▼                           ▼
┌─────────────────┐     ┌─────────────────┐         ┌─────────────────┐
│  Document       │     │  Language       │         │  Conversation   │
│  Storage        │     │  Detection      │         │  Memory         │
└─────────────────┘     └─────────────────┘         └─────────────────┘
```

## Components

### Frontend (React)
- **State Management**: React Context and hooks
- **Key Features**:
  - PDF upload with drag-and-drop
  - Real-time chat interface
  - Document preview
  - Multi-language support
- **Communication**: REST API and WebSocket

### Backend (FastAPI)
- **API Layer**: RESTful endpoints for PDF operations and chat
- **PDF Processing**: Text extraction, OCR, language detection
- **Vector Store**: FAISS for semantic search and context retrieval
- **Chat Management**: Session handling and conversation history
- **Language Processing**: Multi-language support and detection

## Data Flow

### Document Processing
1. PDF upload → Text extraction → OCR (if needed)
2. Language detection → Text chunking → Vector embedding
3. Store in FAISS vector database

### Chat Flow
1. User question → Context retrieval from vector store
2. Format prompt with context → Send to LLM (Ollama)
3. Generate response → Update chat history → Return to user

## External Services

### Ollama (Local LLM)
- URL: `http://localhost:11434`
- Model: Mistral
- Purpose: Question answering and response generation

### Tesseract OCR
- Purpose: Image text extraction
- Support: Multi-language OCR

## Security & Performance

### Security
- File type validation and size limits
- CORS configuration
- Local data storage only

### Performance
- Efficient PDF processing pipeline
- Vector store optimization
- Stateless API design

## File Structure
```
backend/
├── config/          # Configuration management
├── data/            # PDF storage and embeddings
├── utils/           # Core services
│   ├── api.py       # API endpoints
│   ├── chat_manager.py
│   ├── pdf_processor.py
│   └── rag_engine.py
└── run.py           # Application entry point

frontend/
├── src/
│   ├── App.js       # Main application
│   └── styles/      # CSS styling
└── public/          # Static assets
``` 