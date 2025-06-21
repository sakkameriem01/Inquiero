"""
Pydantic models for the Inquiero API.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional

class Message(BaseModel):
    """Chat message data model."""
    text: str
    sender: str
    language: Optional[str] = 'en'

class ChatResponse(BaseModel):
    """Response model for chat replies."""
    response: str
    chat_id: str

class LanguageDetectionRequest(BaseModel):
    """Request model for language detection."""
    text: str

class LanguageDetectionResponse(BaseModel):
    """Response model for detected language."""
    language: str

class RenameChatRequest(BaseModel):
    """Request model for renaming a chat."""
    name: str 