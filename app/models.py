from typing import List, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_id: int
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int
