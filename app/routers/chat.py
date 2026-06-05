from fastapi import APIRouter, Depends
from app.models.chat import ChatRequest
from app.services.chat_service import ChatService, get_chat_service

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/")
async def chat(request: ChatRequest, service: ChatService = Depends(get_chat_service)):
    return await service.chat(request)

@router.get("/")
async def list_conversations(service: ChatService = Depends(get_chat_service)):
    return await service.list_conversations()

@router.get("/{chat_id}")
async def get_conversation(chat_id: str, service: ChatService = Depends(get_chat_service)):
    return await service.get_conversation(chat_id)

@router.delete("/{chat_id}")
async def delete_conversation(chat_id: str, service: ChatService = Depends(get_chat_service)):
    return await service.delete_conversation(chat_id)
