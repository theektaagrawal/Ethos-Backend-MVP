import httpx
import json
import logging
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from app.models.chat import ChatRequest
from app.services.persona_service import get_persona_service
from app.services.openrag_client import get_openrag_client
from app.services.langflow_client import get_langflow_client

logger = logging.getLogger(__name__)

# The Langflow Agent component ID whose system_prompt we override
AGENT_COMPONENT_ID = "Agent-Nfw7u"


class ChatService:
    def __init__(self):
        self.persona_service = get_persona_service()
        self.openrag_client = get_openrag_client()
        self.langflow_client = get_langflow_client()

    async def chat(self, request: ChatRequest):
        """
        Route chats directly through OpenRAG's public /v1/chat endpoint without persona system prompt modifications.
        """
        message_to_send = request.message

        if request.stream:
            async def stream_generator():
                try:
                    async with self.openrag_client.client.stream(
                        "POST",
                        "/v1/chat",
                        json={
                            "message": message_to_send,
                            "stream": True,
                            "chat_id": request.chat_id,
                        }
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            yield f"{line}\n"
                except Exception as e:
                    logger.error(f"OpenRAG stream error: {e}")
                    yield f"data: {json.dumps({'error': f'OpenRAG error: {str(e)}'})}\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            try:
                response = await self.openrag_client.client.post(
                    "/v1/chat",
                    json={
                        "message": message_to_send,
                        "stream": False,
                        "chat_id": request.chat_id,
                    }
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "response": data.get("response", ""),
                    "chat_id": data.get("chat_id") or request.chat_id,
                }
            except Exception as e:
                logger.error(f"OpenRAG non-stream error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def list_conversations(self):
        response = await self.openrag_client.client.get("/v1/chat")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="OpenRAG error")
        return response.json()

    async def get_conversation(self, chat_id: str):
        response = await self.openrag_client.client.get(f"/v1/chat/{chat_id}")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="OpenRAG error")
        return response.json()

    async def delete_conversation(self, chat_id: str):
        response = await self.openrag_client.client.delete(f"/v1/chat/{chat_id}")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="OpenRAG error")
        return response.json()

_chat_service_instance = None

def get_chat_service() -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
