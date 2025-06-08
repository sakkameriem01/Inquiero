from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Message(BaseModel):
    content: str
    type: str  # 'user', 'assistant', 'system', 'error'
    timestamp: datetime = datetime.now()

class ChatSession(BaseModel):
    id: str
    title: str
    tags: List[str]
    messages: List[Message]
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    files: List[str]  # List of processed PDF filenames 