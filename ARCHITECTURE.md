# Inquiero System Architecture

## System Overview
Inquiero is a document-based Q&A system that processes PDF documents and enables interactive conversations about their content. The system uses a modern microservices architecture with a FastAPI backend and a React frontend.

## Architecture Diagram
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

## Component Details

### Frontend (React)
- **State Management**: Uses React Context and hooks for state management
- **Key Components**:
  - PDF Upload Component
  - Chat Interface
  - Document Viewer
  - Language Selection
- **Communication**: 
  - REST API calls for document operations
  - WebSocket for real-time chat
  - File upload using multipart/form-data

### Backend (FastAPI)
1. **API Layer** (`app.py`, `api.py`)
   - RESTful endpoints for:
     - PDF upload and management
     - Chat operations
     - Language detection
   - WebSocket endpoints for real-time chat
   - CORS middleware for frontend communication

2. **PDF Processing Service** (`pdf_processor.py`)
   - Document processing pipeline:
     ```
     PDF Upload → Text Extraction → OCR (if needed) → 
     Language Detection → Text Chunking → Vector Embedding
     ```
   - Features:
     - PDF text extraction using PyMuPDF
     - OCR using Tesseract
     - Language detection
     - Text chunking for better context

3. **Vector Store** (FAISS)
   - Stores document embeddings
   - Enables semantic search
   - Maintains document context for Q&A

4. **Chat Management** (`chat_manager.py`)
   - Manages chat sessions
   - Maintains conversation history
   - Integrates with LLM (Ollama/Mistral)

5. **Language Processing**
   - Language detection using langdetect
   - Translation capabilities using googletrans
   - Multi-language support

## Data Flow

### Document Processing Flow
1. User uploads PDF
2. Backend receives file
3. PDF Processor:
   - Extracts text
   - Performs OCR if needed
   - Detects language
   - Creates text chunks
   - Generates embeddings
4. Vector store is updated
5. Document is ready for Q&A

### Chat Flow
1. User sends question
2. Backend:
   - Retrieves relevant context from vector store
   - Formats prompt with context
   - Sends to LLM (Ollama)
3. Response is generated
4. Chat history is updated
5. Response sent to frontend

## External Services Integration

### Ollama Integration
- Local LLM service
- Default URL: `http://localhost:11434`
- Uses Mistral model
- Handles:
  - Question answering
  - Context understanding
  - Response generation

### Tesseract OCR
- Local OCR service
- Handles:
  - Image text extraction
  - Multi-language OCR
  - PDF image processing

## Security Considerations
1. **File Upload Security**
   - File type validation
   - Size limits
   - Content scanning

2. **API Security**
   - CORS configuration
   - Rate limiting
   - Input validation

3. **Data Security**
   - Local storage only
   - No sensitive data transmission
   - Secure file handling

## Scalability Considerations
1. **Backend Scalability**
   - Stateless API design
   - Efficient PDF processing
   - Vector store optimization

2. **Frontend Scalability**
   - Component-based architecture
   - Efficient state management
   - Lazy loading

3. **Resource Management**
   - PDF storage cleanup
   - Chat history management
   - Memory optimization

## Error Handling
1. **Frontend Error Handling**
   - User-friendly error messages
   - Graceful degradation
   - Retry mechanisms

2. **Backend Error Handling**
   - Proper HTTP status codes
   - Detailed error logging
   - Exception handling

## Monitoring and Logging
1. **Backend Logging**
   - PDF processing logs
   - API request logs
   - Error logs

2. **Performance Monitoring**
   - Response times
   - Resource usage
   - Error rates 