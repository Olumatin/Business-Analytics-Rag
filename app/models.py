from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4

class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_id: int
    score: Optional[float] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int
