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
from sqlalchemy.orm import Session as DBSession, joinedload
from database import get_db, engine
from models import Base
import json

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
    files: Optional[List[Dict[str, str]]] = None  # List of file info dictionaries

def generate_title_from_question(question: str) -> str:
    """Generate a title from the first user question."""
    # Take first 50 characters or up to the first sentence
    title = question[:50].strip()
    if len(question) > 50:
        title += "..."
    return title

# Global exception handler for uncaught exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail=f"Internal server error: {exc}").response()

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
        logger.info(f"Received file upload request for: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = file.filename
        filename = f"{timestamp}_{original_filename}"
        
        # Save the file
        file_path = os.path.join("uploads", filename)
        logger.info(f"Saving file to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved successfully: {filename}")

        # Create new session if session_id is not provided
        if not session_id:
            db_session = Session(
                title=f"Chat with {original_filename}",
                tags=[],
                messages=[]
            )
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
            session_id = db_session.id
            logger.info(f"Created new session: {session_id}")
        else:
            db_session = db.query(Session).filter(Session.id == session_id).first()
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found")
            logger.info(f"Using existing session: {session_id}")

        # Process the PDF with session-specific vector store
        try:
            result = pdf_processor.process_pdf(file_path, filename, str(session_id))
            chunks_count = result["chunks"]
            pages_count = result["pages"]
            text_length = result["text_length"]
            
            logger.info(f"PDF processing completed successfully:")
            logger.info(f"- Pages processed: {pages_count}")
            logger.info(f"- Text extracted: {text_length} characters")
            logger.info(f"- Chunks created: {chunks_count}")

            # Initialize QA chain with the session-specific vector store
            vector_store = pdf_processor.get_session_vector_store(str(session_id))
            if not vector_store:
                raise HTTPException(status_code=500, detail="Failed to create vector store for session")
            
            qa_system.create_qa_chain(vector_store, str(session_id))
            logger.info(f"Successfully initialized QA system for session {session_id}")

            # Save file info to database
            db_file = DBFile(
                filename=filename,
                original_filename=original_filename,
                chunks=chunks_count,
                pages=pages_count
            )
            db.add(db_file)
            db.flush()  # Get the file ID without committing

            # Link the file to the session
            db_session.files.append(db_file)
            db.commit()

            return {
                "message": "File uploaded and processed successfully",
                "filename": filename,
                "original_filename": original_filename,
                "chunks": chunks_count,
                "pages": pages_count,
                "text_length": text_length,
                "session_id": session_id,
                "file_id": db_file.id,
                "processing_details": {
                    "pages_processed": pages_count,
                    "text_extracted": text_length,
                    "chunks_created": chunks_count,
                    "vector_store_created": True,
                    "qa_system_initialized": True
                }
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            # Clean up the uploaded file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(request: ChatRequest = Body(...), db: DBSession = Depends(get_db)):
    """Get answer for a question and manage chat session."""
    try:
        db_session = None
        if request.session_id:
            db_session = db.query(Session).options(joinedload(Session.files)).filter(Session.id == request.session_id).first()
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            # If no session_id, create a new session
            db_session = Session(
                title=request.question[:50] + "..." if len(request.question) > 50 else request.question,
                tags=request.tags or [],
                messages=[]
            )
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
            logger.info(f"Created new session {db_session.id} for chat request without session_id.")

            # If this new session was created because the first chat message came with files
            # (i.e., frontend didn't have session_id yet), link those files to this new session.
            if request.files:
                for file_info in request.files:
                    db_file = db.query(DBFile).filter(DBFile.id == int(file_info['id'])).first()
                    if db_file and db_file not in db_session.files:
                        db_session.files.append(db_file)
                        logger.info(f"Linked file {db_file.filename} to new session {db_session.id}")
                db.commit()
                db.refresh(db_session)

        logger.info(f"Session {db_session.id} retrieved/created. Files linked: {[f.filename for f in db_session.files]}")

        # Check if QA system is initialized for this session
        # Always attempt to initialize if not already, using files linked to the session
        if not qa_system.is_initialized(str(db_session.id)):
            if db_session.files:
                logger.info(f"Attempting to initialize QA system for session {db_session.id} with {len(db_session.files)} files.")
                try:
                    for file in db_session.files:
                        file_path = os.path.join("uploads", file.filename)
                        if os.path.exists(file_path):
                            pdf_processor.process_pdf(file_path, file.filename, str(db_session.id))
                            logger.info(f"Re-processed file {file.filename} for session {db_session.id}")

                    vector_store = pdf_processor.get_session_vector_store(str(db_session.id))
                    if vector_store:
                        qa_system.create_qa_chain(vector_store, str(db_session.id))
                        logger.info(f"Successfully initialized QA system for session {db_session.id}")
                    else:
                        raise ValueError("Failed to create vector store for session.")
                except Exception as e:
                    logger.error(f"Error initializing QA system with session files for session {db_session.id}: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to initialize QA system with session files: {str(e)}"
                    )
            else:
                logger.warning(f"QA system not initialized for session {db_session.id} because no files are linked.")
                raise HTTPException(
                    status_code=400,
                    detail="No files associated with this session. Please upload a PDF file first to initialize the QA system."
                )

        try:
            # Get answer from QA system using session-specific chain
            result = qa_system.get_answer(request.question, str(db_session.id))
            
            # Ensure the answer is a string
            answer = result.get("answer", "")
            if not isinstance(answer, str):
                if isinstance(answer, dict):
                    answer = json.dumps(answer, ensure_ascii=False, indent=2)
                else:
                    answer = str(answer)
            
            # Create new messages
            new_messages = [
                {"type": "user", "content": request.question},
                {"type": "assistant", "content": answer}
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
                "answer": answer,
                "session": {
                    "id": db_session.id,
                    "title": db_session.title,
                    "tags": db_session.tags,
                    "messages": db_session.messages,
                    "created_at": db_session.created_at.isoformat(),
                    "updated_at": db_session.updated_at.isoformat(),
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
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting answer from QA system: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate answer. Please try again: {str(e)}")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in the chat endpoint: {str(e)}")

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
    """Delete a chat session and its associated data."""
    try:
        # Find the session in the database
        db_session = db.query(Session).filter(Session.id == session_id).first()
        if not db_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Clear session-specific data from PDF processor and QA system
        pdf_processor.clear_session(str(session_id))
        qa_system.clear_session(str(session_id))
        
        # Delete the session
        db.delete(db_session)
        db.commit()
        
        return {"message": "Chat session deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat-sessions/")
async def delete_all_chat_sessions(db: DBSession = Depends(get_db)):
    """Delete all chat sessions and their associated data."""
    try:
        # Get all sessions
        db_sessions = db.query(Session).all()
        
        # Clear session-specific data from PDF processor and QA system for each session
        for session in db_sessions:
            pdf_processor.clear_session(str(session.id))
            qa_system.clear_session(str(session.id))
        
        # Delete all sessions
        for session in db_sessions:
            db.delete(session)
        
        db.commit()
        
        return {"message": "All chat sessions deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting all chat sessions: {str(e)}")
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
    """Clear all processed files and vector stores."""
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
        
        # Clear all vector stores and QA chains
        pdf_processor.clear_all()
        qa_system.clear_all()
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