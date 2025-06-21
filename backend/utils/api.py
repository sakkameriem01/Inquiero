"""
FastAPI application and route definitions for Inquiero backend.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import logging
from typing import Dict, Optional
from pathlib import Path

from .pdf_processor import PDFProcessor
from .chat_manager import ChatManager
from .models import Message, LanguageDetectionRequest

logger = logging.getLogger(__name__)

pdf_processor = PDFProcessor()
chat_manager = ChatManager()

app = FastAPI(title="Inquiero API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), chat_id: Optional[str] = Body(None)):
    """Upload a PDF file and process it. Optionally append to an existing chat."""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        content = await file.read()
        pdf_path = pdf_processor.save_pdf(content, file.filename)
        text_content, metadata = pdf_processor.process_pdf(pdf_path)
        if chat_id:
            chat = chat_manager.get_chat(chat_id)
            if not chat:
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
    """Create a new, empty chat session."""
    try:
        chat_data = chat_manager.create_new_chat()
        return JSONResponse(chat_data, status_code=201)
    except Exception as e:
        logger.error(f"Error creating new chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create new chat")

@app.get("/chats/")
async def get_chats():
    """Get all available chats."""
    try:
        chats = chat_manager.get_all_chats()
        return JSONResponse({"chats": chats})
    except Exception as e:
        logger.error(f"Error getting chats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}/")
async def get_chat(chat_id: str):
    """Get a specific chat by ID."""
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
    """Send a message to a chat and get an AI response."""
    try:
        chat = chat_manager.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        user_text = message.text
        if not user_text:
            raise HTTPException(status_code=400, detail="Message text is required")
        language = message.language or "en"
        chat_manager.add_message(chat_id, "user", user_text)
        pdf_paths = chat.get("pdf_paths", [])
        chat_history = chat.get("messages", [])
        ai_response = pdf_processor.get_llm_response(user_text, pdf_paths, chat_history, language=language)
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
    """Delete a chat by ID."""
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
    """Delete all chats."""
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
    """Remove a PDF from a specific chat."""
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
    """Rename a chat."""
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
    """Detect the language of a text."""
    try:
        language = pdf_processor.detect_language(request.text)
        return JSONResponse({"language": language})
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/stats/")
async def get_rag_stats():
    """Get RAG system statistics and performance metrics."""
    try:
        stats = pdf_processor.get_rag_stats()
        return JSONResponse({"rag_stats": stats})
    except Exception as e:
        logger.error(f"Error getting RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/clear-memory/")
async def clear_rag_memory():
    """Clear the RAG engine's conversation memory."""
    try:
        pdf_processor.clear_rag_memory()
        return JSONResponse({"message": "RAG memory cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing RAG memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({"status": "healthy", "service": "Inquiero RAG API"})

pdfs_dir = Path("data/pdfs")
if pdfs_dir.exists():
    app.mount("/pdfs", StaticFiles(directory=str(pdfs_dir)), name="pdfs")

@app.get("/pdf/{pdf_path:path}")
async def serve_pdf(pdf_path: str):
    """Serve a PDF file for preview."""
    try:
        full_path = pdfs_dir / pdf_path
        if not str(full_path.resolve()).startswith(str(pdfs_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")
        if not full_path.suffix.lower() == '.pdf':
            raise HTTPException(status_code=400, detail="File is not a PDF")
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

@app.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    """WebSocket endpoint for real-time chat communication."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            chat = chat_manager.get_chat(chat_id)
            if not chat:
                await websocket.send_text(json.dumps({"error": "Chat not found"}))
                continue
            user_text = message_data.get("text", "")
            chat_manager.add_message(chat_id, "user", user_text)
            pdf_paths = chat.get("pdf_paths", [])
            chat_history = chat.get("messages", [])
            ai_response = pdf_processor.get_llm_response(user_text, pdf_paths, chat_history)
            chat_manager.add_message(chat_id, "assistant", ai_response)
            await websocket.send_text(json.dumps({
                "response": ai_response,
                "chat_id": chat_id
            }))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for chat {chat_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_text(json.dumps({"error": str(e)})) 