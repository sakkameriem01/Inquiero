from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
from utils.pdf_loader import PDFProcessor
from utils.qa_chain import QASystem
from models import ChatSession, Message
import tempfile
import shutil
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PDF processor and QA system
pdf_processor = PDFProcessor()
qa_system = QASystem()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory storage for chat sessions (replace with database in production)
chat_sessions: Dict[str, ChatSession] = {}

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None

@app.get("/")
async def root():
    return {"message": "Welcome to Inquiero API"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF
        try:
            result = pdf_processor.process_pdf(str(file_path), file.filename)
            
            # Update QA system with processed files info
            qa_system.update_processed_files(pdf_processor.get_processed_files())
            
            # Create QA chain if not exists
            if not qa_system.qa_chain:
                qa_system.create_qa_chain(pdf_processor.vector_store)
            
            return {
                "message": "File processed successfully",
                "filename": file.filename,
                "chunks": result["chunks"],
                "pages": result["pages"]
            }
        finally:
            # Clean up the temporary file
            if file_path.exists():
                file_path.unlink()
                
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(request: ChatRequest):
    """Get answer for a question and manage chat session."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Create or get chat session
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatSession(
                id=session_id,
                title=request.question[:50] + "..." if len(request.question) > 50 else request.question,
                tags=request.tags or [],
                messages=[],
                files=list(pdf_processor.get_processed_files().keys())
            )
        
        # Add user message
        user_message = Message(content=request.question, type="user")
        chat_sessions[session_id].messages.append(user_message)
        
        # Get answer from QA system
        result = qa_system.get_answer(request.question)
        
        # Add assistant message
        assistant_message = Message(content=result["answer"], type="assistant")
        chat_sessions[session_id].messages.append(assistant_message)
        
        # Update session timestamp
        chat_sessions[session_id].updated_at = datetime.now()
        
        return {
            "session_id": session_id,
            "answer": result["answer"],
            "sources": result.get("sources", [])
        }
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-sessions/")
async def get_chat_sessions(
    tag: Optional[str] = None,
    search: Optional[str] = None
):
    """Get list of chat sessions with optional filtering."""
    try:
        sessions = list(chat_sessions.values())
        
        # Filter by tag if provided
        if tag:
            sessions = [s for s in sessions if tag in s.tags]
        
        # Filter by search term if provided
        if search:
            search = search.lower()
            sessions = [s for s in sessions if 
                       search in s.title.lower() or 
                       any(search in tag.lower() for tag in s.tags)]
        
        # Sort by updated_at
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error getting chat sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get a specific chat session."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return chat_sessions[session_id]
    except Exception as e:
        logger.error(f"Error getting chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/chat-sessions/{session_id}")
async def update_chat_session(
    session_id: str,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None
):
    """Update chat session title or tags."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        session = chat_sessions[session_id]
        if title:
            session.title = title
        if tags is not None:
            session.tags = tags
        session.updated_at = datetime.now()
        
        return session
    except Exception as e:
        logger.error(f"Error updating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat-sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        del chat_sessions[session_id]
        return {"message": "Chat session deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/")
async def get_processed_files():
    """Get list of processed files."""
    try:
        return {
            "files": pdf_processor.get_processed_files()
        }
    except Exception as e:
        logger.error(f"Error getting processed files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/")
async def clear_files():
    """Clear all processed files and vector store."""
    try:
        pdf_processor.clear_vector_store()
        qa_system.qa_chain = None
        qa_system.processed_files = {}
        return {"message": "All files cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 