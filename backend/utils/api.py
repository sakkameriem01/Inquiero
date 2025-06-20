"""
API Module

This module contains the FastAPI application and route definitions.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from .pdf_processor import PDFProcessor
from .chat_manager import ChatManager
from .models import Message, LanguageDetectionRequest

# Configure logging
logger = logging.getLogger(__name__)

# Initialize components
pdf_processor = PDFProcessor()
chat_manager = ChatManager()

# Create FastAPI app
app = FastAPI(title="Inquiero API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), chat_id: Optional[str] = Body(None)):
    """
    Upload a PDF file for processing. If chat_id is provided, appends to existing
    chat. Otherwise, creates a new chat.
    
    Args:
        file: PDF file to upload
        chat_id: Optional chat ID to append the PDF to
        
    Returns:
        JSON response with upload status and file information
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        content = await file.read()
        pdf_path = pdf_processor.save_pdf(content, file.filename)
        text_content, metadata = pdf_processor.process_pdf(pdf_path)
        
        if chat_id:
            chat = chat_manager.get_chat(chat_id)
            if not chat:
                # This case can be handled more gracefully, maybe create a new one?
                # For now, we'll assume chat_id is valid if provided.
                raise HTTPException(status_code=404, detail="Chat not found")
            
            updated_chat = chat_manager.append_pdf_to_chat(chat_id, pdf_path)
            if not updated_chat:
                raise HTTPException(status_code=500, detail="Failed to append PDF to chat")

            return JSONResponse({
                "message": "PDF uploaded and appended to chat successfully",
                "filename": file.filename,
                "pdf_path": pdf_path,
            "chat_id": chat_id,
                "chat_name": updated_chat.get("name", "Chat"),
                "language": metadata.get("language", "en"),
                "metadata": metadata
            })
        else:
            # Create a new chat for this PDF
            chat_data = chat_manager.create_new_chat(pdf_path=pdf_path)
            
            return JSONResponse({
                "message": "PDF uploaded and a new chat was created successfully",
            "filename": file.filename,
                "pdf_path": pdf_path,
                "chat_id": chat_data["id"],
                "chat_name": chat_data["name"],
                "language": metadata.get("language", "en"),
                "metadata": metadata
            })
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats")
async def create_new_chat_endpoint():
    """
    Create a new, empty chat session.
    """
    try:
        chat_data = chat_manager.create_new_chat()
        return JSONResponse(chat_data, status_code=201)
    except Exception as e:
        logger.error(f"Error creating new chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create new chat")

@app.get("/chats/")
async def get_chats():
    """
    Get all available chats.
    
    Returns:
        List of chat objects
    """
    try:
        chats = chat_manager.get_all_chats()
        return JSONResponse({"chats": chats})
    except Exception as e:
        logger.error(f"Error getting chats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}/")
async def get_chat(chat_id: str):
    """
    Get a specific chat by ID.
    
    Args:
        chat_id: Unique identifier for the chat
        
    Returns:
        Chat object
    """
    try:
        chat = chat_manager.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return JSONResponse(chat)
    except Exception as e:
        logger.error(f"Error getting chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats/{chat_id}/message/")
async def send_message(chat_id: str, message: Message):
    """
    Send a message to a chat and get AI response.
    
    Args:
        chat_id: Unique identifier for the chat
        message: Message object containing text
        
    Returns:
        AI response
    """
    try:
        # Get chat
    chat = chat_manager.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Add user message to chat
        user_text = message.text
        if not user_text:
            raise HTTPException(status_code=400, detail="Message text is required")
        
        language = message.language or "en"
        chat_manager.add_message(chat_id, "user", user_text)

        # Get PDF paths for this chat
    pdf_paths = chat.get("pdf_paths", [])
        
        # Get chat history
        chat_history = chat.get("messages", [])
        
        # Generate AI response using RAG
        ai_response = pdf_processor.get_llm_response(user_text, pdf_paths, chat_history, language=language)

        # Add AI response to chat
        chat_manager.add_message(chat_id, "assistant", ai_response)
        
        return JSONResponse({
            "response": ai_response,
            "chat_id": chat_id
        })
        
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}/")
async def delete_chat(chat_id: str):
    """
    Delete a chat.
    
    Args:
        chat_id: Unique identifier for the chat
        
    Returns:
        Success message
    """
    try:
        success = chat_manager.delete_chat(chat_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
        return JSONResponse({"message": "Chat deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/all/")
async def delete_all_chats():
    """
    Delete all chats.
    
    Returns:
        Success message
    """
    try:
        success = chat_manager.delete_all_chats()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete all chats")
        return JSONResponse({"message": "All chats deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting all chats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}/pdfs/")
async def remove_pdf_from_chat(chat_id: str, payload: Dict = Body(...)):
    """
    Remove a PDF from a specific chat.
    
    Args:
        chat_id: The ID of the chat to modify.
        payload: A dictionary containing the 'pdf_path' to remove.
        
    Returns:
        A success message.
    """
    logger.info(f"Attempting to remove PDF from chat_id: {chat_id}")
    logger.info(f"Received payload: {payload}")
    pdf_path = payload.get("pdf_path")
    if not pdf_path:
        logger.error("Request failed: pdf_path not in payload")
        raise HTTPException(status_code=400, detail="PDF path is required")

    try:
        success = chat_manager.remove_pdf_from_chat(chat_id, pdf_path)
        if not success:
            logger.warning(f"remove_pdf_from_chat returned False for chat {chat_id} and path {pdf_path}")
            raise HTTPException(status_code=404, detail="Chat or PDF not found in chat")
        
        logger.info(f"Successfully removed {pdf_path} from chat {chat_id}")
        return JSONResponse({"message": "PDF removed from chat successfully"})
    except Exception as e:
        logger.error(f"Error removing PDF from chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/chats/{chat_id}/rename/")
async def rename_chat(chat_id: str, rename_data: Dict):
    """
    Rename a chat.
    
    Args:
        chat_id: Unique identifier for the chat
        rename_data: Object containing new name
        
    Returns:
        Success message
    """
    try:
        new_name = rename_data.get("name", "")
        if not new_name:
            raise HTTPException(status_code=400, detail="New name is required")
        
        success = chat_manager.rename_chat(chat_id, new_name)
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return JSONResponse({"message": "Chat renamed successfully"})
    except Exception as e:
        logger.error(f"Error renaming chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-language/")
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect the language of a text.
    
    Args:
        request: LanguageDetectionRequest containing text
        
    Returns:
        LanguageDetectionResponse with detected language
    """
    try:
        language = pdf_processor.detect_language(request.text)
        return JSONResponse({"language": language})
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/stats/")
async def get_rag_stats():
    """
    Get RAG system statistics and performance metrics.
    
    Returns:
        RAG system statistics
    """
    try:
        stats = pdf_processor.get_rag_stats()
        return JSONResponse({"rag_stats": stats})
    except Exception as e:
        logger.error(f"Error getting RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/clear-memory/")
async def clear_rag_memory():
    """
    Clear the RAG engine's conversation memory.
    
    Returns:
        Success message
    """
    try:
        pdf_processor.clear_rag_memory()
        return JSONResponse({"message": "RAG memory cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing RAG memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return JSONResponse({"status": "healthy", "service": "Inquiero RAG API"})

# Mount static files for PDF serving
pdfs_dir = Path("data/pdfs")
if pdfs_dir.exists():
    app.mount("/pdfs", StaticFiles(directory=str(pdfs_dir)), name="pdfs")

@app.get("/pdf/{pdf_path:path}")
async def serve_pdf(pdf_path: str):
    """
    Serve a PDF file for preview.
    
    Args:
        pdf_path: Path to the PDF file relative to the PDFs directory
        
    Returns:
        PDF file response
    """
    try:
        # Construct the full path to the PDF file
        full_path = pdfs_dir / pdf_path
        
        # Security check: ensure the path is within the PDFs directory
        if not str(full_path.resolve()).startswith(str(pdfs_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        # Check if it's actually a PDF file
        if not full_path.suffix.lower() == '.pdf':
            raise HTTPException(status_code=400, detail="File is not a PDF")
        
        # Return the PDF file
        return FileResponse(
            path=str(full_path),
            media_type='application/pdf',
            filename=full_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving PDF {pdf_path}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve PDF file")

# WebSocket endpoint for real-time chat
@app.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    """
    WebSocket endpoint for real-time chat communication.
    
    Args:
        websocket: WebSocket connection
        chat_id: Unique identifier for the chat
    """
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Get chat
            chat = chat_manager.get_chat(chat_id)
            if not chat:
                await websocket.send_text(json.dumps({
                    "error": "Chat not found"
                }))
                continue
            
            # Add user message
            user_text = message_data.get("text", "")
            chat_manager.add_message(chat_id, "user", user_text)
            
            # Get PDF paths and chat history
            pdf_paths = chat.get("pdf_paths", [])
            chat_history = chat.get("messages", [])
            
            # Generate AI response
            ai_response = pdf_processor.get_llm_response(user_text, pdf_paths, chat_history)
            
            # Add AI response to chat
            chat_manager.add_message(chat_id, "assistant", ai_response)
            
            # Send response back to client
            await websocket.send_text(json.dumps({
                "response": ai_response,
                "chat_id": chat_id
            }))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for chat {chat_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_text(json.dumps({
            "error": str(e)
        })) 