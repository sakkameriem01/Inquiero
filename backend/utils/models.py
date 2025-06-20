"""
Pydantic models for the Inquiero API.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional

class Message(BaseModel):
    """Data model for a chat message."""
    text: str
    sender: str
    language: Optional[str] = 'en'

class ChatResponse(BaseModel):
    response: str
    chat_id: str

class LanguageDetectionRequest(BaseModel):
    """Request model for language detection."""
    text: str

class LanguageDetectionResponse(BaseModel):
    language: str

class RenameChatRequest(BaseModel):
    name: str 