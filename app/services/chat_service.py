import httpx
import json
import asyncio
import os
import openai
import logging
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from app.models.chat import ChatRequest
from app.services.persona_service import get_persona_service
from app.services.openrag_client import get_openrag_client
from app.services.langflow_client import get_langflow_client
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# The Langflow Agent component ID whose system_prompt we override
AGENT_COMPONENT_ID = "Agent-Nfw7u"


class ChatService:
    def __init__(self):
        self.persona_service = get_persona_service()
        self.openrag_client = get_openrag_client()
        self.langflow_client = get_langflow_client()

    async def chat(self, request: ChatRequest):
        persona = self.persona_service.get_persona(request.persona_id)
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")

        if persona.id == "p_os":
            return await self._handle_os_chat(request, persona)
        else:
            return await self._handle_persona_chat(request, persona)

    async def _handle_persona_chat(self, request: ChatRequest, persona):
        """
        Route persona chats through OpenRAG's public /v1/chat endpoint.
        Uses prepended system directives to adopts the custom persona.
        """
        # Prepend system prompt directive to user message to adoption of custom persona
        message_to_send = request.message
        if persona.system_prompt:
            message_to_send = (
                f"[SYSTEM DIRECTIVE]\n"
                f"You must adopt the following persona instructions for this turn:\n"
                f"{persona.system_prompt}\n"
                f"[END OF SYSTEM DIRECTIVE]\n\n"
                f"User Message: {request.message}"
            )

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

    async def _handle_os_chat(self, request: ChatRequest, os_persona):
        """
        Call OS: fan-out to all personas via OpenRAG calls,
        then synthesize with OpenAI.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required for Call OS synthesis")
        openai_client = openai.AsyncOpenAI(api_key=api_key)

        personas = [p for p in self.persona_service.list_personas() if p.id != "p_os"]
        
        async def fetch_persona_response(p):
            """Call OpenRAG directly for each persona with its system prompt prepended."""
            message_to_send = request.message
            if p.system_prompt:
                message_to_send = (
                    f"[SYSTEM DIRECTIVE]\n"
                    f"You must adopt the following persona instructions for this turn:\n"
                    f"{p.system_prompt}\n"
                    f"[END OF SYSTEM DIRECTIVE]\n\n"
                    f"User Message: {request.message}"
                )

            try:
                response = await self.openrag_client.client.post(
                    "/v1/chat",
                    json={
                        "message": message_to_send,
                        "stream": False,
                    }
                )
                response.raise_for_status()
                data = response.json()
                return p.name, data.get("response", "")
            except Exception as e:
                logger.warning(f"OpenRAG call failed for {p.name}: {e}")
                return p.name, "No perspective available."

        # Fetch all in parallel
        results = await asyncio.gather(*(fetch_persona_response(p) for p in personas))
        
        synthesize_prompt = "You are the OS, representing a synthesized board of directors. The user asked a question. Here are the perspectives of the board members:\n\n"
        for name, text in results:
            synthesize_prompt += f"--- {name} ---\n{text}\n\n"
        synthesize_prompt += "Your task: Synthesize these perspectives into a final, structured response. Summarize what the board members agree or disagree on, and provide a final recommendation. Format it exactly as required by the OS persona."

        if request.stream:
            async def os_stream_generator():
                try:
                    stream_response = await openai_client.chat.completions.create(
                        model="gpt-5-mini",
                        messages=[
                            {"role": "system", "content": os_persona.system_prompt},
                            {"role": "user", "content": synthesize_prompt}
                        ],
                        stream=True
                    )
                    async for chunk in stream_response:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            yield f"data: {json.dumps({'type': 'content', 'delta': content})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'chat_id': request.chat_id})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'OS synthesis error: {str(e)}'})}\n\n"

            return StreamingResponse(
                os_stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            try:
                response = await openai_client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": os_persona.system_prompt},
                        {"role": "user", "content": synthesize_prompt}
                    ]
                )
                return {"response": response.choices[0].message.content, "chat_id": request.chat_id}
            except Exception as e:
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
