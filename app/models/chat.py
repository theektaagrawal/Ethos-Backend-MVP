from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class ChatRequest(BaseModel):
    message: str
    persona_id: str
    chat_id: Optional[str] = None
    stream: bool = True
    filters: Optional[Dict[str, Any]] = None

class ChatSource(BaseModel):
    filename: str
    text: str
    score: Optional[float] = None
    page: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    chat_id: Optional[str] = None
    sources: List[ChatSource] = []
