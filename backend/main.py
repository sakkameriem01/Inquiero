from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from dotenv import load_dotenv
from utils.pdf_loader import PDFProcessor
from utils.qa_chain import QASystem
from models import Session, File as DBFile, ChatSession as ChatSessionModel
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session as DBSession
from database import get_db, engine
from models import Base

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
chat_sessions: Dict[str, ChatSessionModel] = {}

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
async def upload_file(
    file: UploadFile = File(),
    session_id: Optional[int] = None,
    db: DBSession = Depends(get_db)
):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = file.filename
        filename = f"{timestamp}_{original_filename}"
        
        # Save the file
        file_path = os.path.join("uploads", filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the PDF
        result = pdf_processor.process_pdf(file_path, filename)
        chunks_count = result["chunks"]
        pages_count = result["pages"]

        # Initialize QA chain with the vector store
        qa_system.create_qa_chain(pdf_processor.vector_store)

        # Save file info to database
        db_file = DBFile(
            filename=filename,
            original_filename=original_filename,
            chunks=chunks_count,
            pages=pages_count
        )
        db.add(db_file)
        db.flush()  # Get the file ID without committing

        # If session_id is provided, link the file to the session
        if session_id:
            session = db.query(Session).filter(Session.id == session_id).first()
            if session:
                session.files.append(db_file)
                db.commit()
            else:
                db.rollback()
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            db.commit()

        return {
            "message": "File uploaded successfully",
            "file_id": db_file.id,
            "filename": filename,
            "original_filename": original_filename,
            "chunks": chunks_count,
            "pages": pages_count
        }

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(request: ChatRequest = Body(...), db: DBSession = Depends(get_db)):
    """Get answer for a question and manage chat session."""
    try:
        # Check if QA chain is initialized
        if not qa_system.qa_chain:
            raise HTTPException(
                status_code=400,
                detail="Please upload a PDF file first to initialize the QA system."
            )

        # Get existing session or create new one only if no session_id provided
        if request.session_id:
            db_session = db.query(Session).filter(Session.id == request.session_id).first()
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            # Create new session only if no session_id provided
            db_session = Session(
                title=request.question[:50] + "..." if len(request.question) > 50 else request.question,
                tags=request.tags or [],
                messages=[]  # Initialize empty messages array
            )
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
        
        try:
            # Get answer from QA system
            result = qa_system.get_answer(request.question, str(db_session.id))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting answer from QA system: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate answer. Please try again.")
        
        # Create new messages
        new_messages = [
            {"type": "user", "content": request.question},
            {"type": "assistant", "content": result["answer"]}
        ]
        
        # Update session with new messages
        if not db_session.messages:
            db_session.messages = new_messages
        else:
            db_session.messages.extend(new_messages)
        
        # Update session title if it's a new session
        if not request.session_id:
            db_session.title = request.question[:50] + "..." if len(request.question) > 50 else request.question
        
        db.commit()
        db.refresh(db_session)
        
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
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/")
async def get_sessions(db: DBSession = Depends(get_db)):
    try:
        sessions = db.query(Session).order_by(Session.updated_at.desc()).all()
        return [
            {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "pinned": session.pinned,
                "files": [
                    {
                        "id": file.id,
                        "filename": file.filename,
                        "original_filename": file.original_filename,
                        "file_type": "pdf",
                        "chunk_count": file.chunks,
                        "page_count": file.pages
                    }
                    for file in session.files
                ]
            }
            for session in sessions
        ]
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
                any(search in msg.content.lower() for msg in s.files)
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
async def get_chat_session(session_id: int, db: DBSession = Depends(get_db)):
    """Get a specific chat session."""
    try:
        logger.info(f"Fetching chat session with ID: {session_id}")
        
        # Query the session from the database
        db_session = db.query(Session).filter(Session.id == session_id).first()
        
        if not db_session:
            logger.warning(f"Session not found with ID: {session_id}")
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Convert the session to a response format
        response = {
            "id": db_session.id,
            "title": db_session.title,
            "tags": db_session.tags,
            "messages": db_session.messages,
            "created_at": db_session.created_at.isoformat(),
            "updated_at": db_session.updated_at.isoformat(),
            "pinned": db_session.pinned,
            "files": [
                {
                    "id": file.id,
                    "filename": file.filename,
                    "original_filename": file.original_filename,
                    "file_type": "pdf",
                    "chunk_count": file.chunks,
                    "page_count": file.pages
                }
                for file in db_session.files
            ]
        }
        
        logger.info(f"Successfully retrieved session: {session_id}")
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting chat session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.put("/chat-sessions/{session_id}")
async def update_chat_session(session_id: int, updates: dict, db: DBSession = Depends(get_db)):
    """Update chat session title, tags, or pinned status."""
    try:
        db_session = db.query(Session).filter(Session.id == session_id).first()
        if not db_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Update fields
        if "title" in updates:
            db_session.title = updates["title"]
        if "tags" in updates:
            db_session.tags = updates["tags"]
        if "pinned" in updates:
            db_session.pinned = updates["pinned"]
        
        db_session.updated_at = datetime.now()
        db.commit()
        db.refresh(db_session)
        
        return {
            "id": db_session.id,
            "title": db_session.title,
            "tags": db_session.tags,
            "messages": db_session.files,
            "created_at": db_session.created_at.isoformat(),
            "updated_at": db_session.updated_at.isoformat(),
            "pinned": db_session.pinned
        }
    except Exception as e:
        logger.error(f"Error updating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat-sessions/{session_id}")
async def delete_chat_session(session_id: int, db: DBSession = Depends(get_db)):
    """Delete a chat session from the database."""
    try:
        # Find the session in the database
        db_session = db.query(Session).filter(Session.id == session_id).first()
        if not db_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Delete the session
        db.delete(db_session)
        db.commit()
        
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
async def clear_files(db: DBSession = Depends(get_db)):
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