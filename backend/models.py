from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Table, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

class Message(BaseModel):
    content: str = Field(..., description="Message content")
    type: str = Field(..., description="Message type (user or assistant)")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatSession(BaseModel):
    id: str
    title: str
    tags: List[str] = Field(default_factory=list)
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    files: List[str] = Field(default_factory=list)
    pinned: bool = Field(default=False)
    auto_title: Optional[str] = None

Base = declarative_base()

# Association table for session-files relationship
session_files = Table(
    'session_files',
    Base.metadata,
    Column('session_id', Integer, ForeignKey('sessions.id')),
    Column('file_id', Integer, ForeignKey('files.id'))
)

class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    tags = Column(JSON, default=list)
    messages = Column(JSON, default=list)
    pinned = Column(Boolean, default=False)
    files = relationship("File", secondary=session_files, back_populates="sessions")

class File(Base):
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    original_filename = Column(String)
    chunks = Column(Integer)
    pages = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("Session", secondary=session_files, back_populates="files") 