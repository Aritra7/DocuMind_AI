# ./backend/app/models.py

import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Enum
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "READY"
    FAILED = "failed"

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    s3_key = Column(String, unique=True)
    status = Column(String, default=DocumentStatus.UPLOADED)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # In a real app, you'd have a owner_id = Column(Integer, ForeignKey("users.id"))
