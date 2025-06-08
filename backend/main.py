from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from dotenv import load_dotenv
from utils.pdf_loader import PDFProcessor, process_pdf
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
from sqlalchemy.orm import Session
import json

from database import get_db, engine
from models import Base, Session as DBSession, File as DBFile

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

# Mount the uploads directory
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Create database tables
Base.metadata.create_all(bind=engine)

# In-memory storage for chat sessions (replace with database in production)
chat_sessions: Dict[str, ChatSession] = {}

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None

def generate_title_from_question(question: str) -> str:
    """Generate a title from the first user question."""
    # Take first 50 characters or up to the first sentence
    title = question[:50].strip()
    if len(question) > 50:
        title += "..."
    return title

@app.get("/")
async def root():
    return {"message": "Welcome to Inquiero API"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
            chunks, pages = process_pdf(str(file_path))
            
            # Save file info to database
            db_file = DBFile(
                filename=file.filename,
                original_filename=file.filename,
                chunks=chunks,
                pages=pages
            )
            db.add(db_file)
            db.commit()
            db.refresh(db_file)
            
            # Update QA system with processed files info
            qa_system.update_processed_files(pdf_processor.get_processed_files())
            
            # Create QA chain if not exists
            if not qa_system.qa_chain:
                qa_system.create_qa_chain(pdf_processor.vector_store)
            
            return {
                "message": "File processed successfully",
                "filename": file.filename,
                "chunks": chunks,
                "pages": pages
            }
        except Exception as e:
            # Clean up the file if processing fails
            if file_path.exists():
                file_path.unlink()
            raise e
                
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(question: str, session_id: Optional[int] = None, tags: List[str] = [], db: Session = Depends(get_db)):
    """Get answer for a question and manage chat session."""
    try:
        # Get or create session
        if session_id:
            db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            db_session = DBSession(
                title=question[:50] + "..." if len(question) > 50 else question,
                tags=tags,
                messages=[]
            )
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
        
        # Get answer from QA system
        result = qa_system.get_answer(question)
        
        # Update session with new messages
        messages = db_session.messages or []
        messages.extend([
            {"type": "user", "content": question},
            {"type": "assistant", "content": result["answer"]}
        ])
        db_session.messages = messages
        db.commit()
        
        return {
            "answer": result["answer"],
            "session": {
                "id": db_session.id,
                "title": db_session.title,
                "tags": db_session.tags,
                "messages": db_session.messages,
                "created_at": db_session.created_at.isoformat(),
                "updated_at": db_session.updated_at.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/")
async def get_sessions(db: Session = Depends(get_db)):
    sessions = db.query(DBSession).all()
    return [
        {
            "id": session.id,
            "title": session.title,
            "tags": session.tags,
            "messages": session.messages,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "files": [
                {
                    "id": file.id,
                    "filename": file.original_filename,
                    "chunks": file.chunks,
                    "pages": file.pages
                }
                for file in session.files
            ]
        }
        for session in sessions
    ]

@app.get("/chat-sessions/")
async def get_chat_sessions(search: Optional[str] = None, tag: Optional[str] = None):
    """Get list of chat sessions with optional filtering."""
    try:
        sessions = list(chat_sessions.values())
        
        # Filter by search term
        if search:
            search = search.lower()
            sessions = [
                s for s in sessions
                if search in s.title.lower() or
                any(search in msg.content.lower() for msg in s.messages)
            ]
        
        # Filter by tag
        if tag:
            sessions = [s for s in sessions if tag in s.tags]
        
        # Sort: pinned first, then by updated_at
        sessions.sort(key=lambda x: (not x.pinned, x.updated_at), reverse=True)
        
        return {"sessions": [s.dict() for s in sessions]}
    except Exception as e:
        logger.error(f"Error getting chat sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get a specific chat session."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return chat_sessions[session_id].dict()
    except Exception as e:
        logger.error(f"Error getting chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/chat-sessions/{session_id}")
async def update_chat_session(session_id: str, updates: dict):
    """Update chat session title or tags."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        session = chat_sessions[session_id]
        
        # Update fields
        if "title" in updates:
            session.title = updates["title"]
        if "tags" in updates:
            session.tags = updates["tags"]
        if "pinned" in updates:
            session.pinned = updates["pinned"]
        
        session.updated_at = datetime.now()
        
        return session.dict()
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
        processed_files = pdf_processor.get_processed_files()
        # Ensure we return a list of filenames
        files_list = list(processed_files.keys()) if isinstance(processed_files, dict) else []
        return {
            "files": files_list
        }
    except Exception as e:
        logger.error(f"Error getting processed files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/")
async def clear_files(db: Session = Depends(get_db)):
    """Clear all processed files and vector store."""
    try:
        # Clear files from database
        db.query(DBFile).delete()
        db.commit()
        
        # Clear files from uploads directory
        for filename in os.listdir(UPLOAD_DIR):
            file_path = UPLOAD_DIR / filename
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        pdf_processor.clear_vector_store()
        qa_system.qa_chain = None
        qa_system.processed_files = {}
        return {"message": "All files cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/uploads/{filename}")
async def get_pdf(filename: str):
    """Serve a PDF file."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 