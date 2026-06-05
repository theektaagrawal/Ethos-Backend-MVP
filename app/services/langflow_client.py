"""
Direct Langflow API client for persona-specific chat routing.
Bypasses OpenRAG and calls Langflow's /api/v1/run endpoint directly,
allowing system prompt tweaks to customize the agent's persona.
"""

import httpx
import json
import logging
from typing import AsyncIterator, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class LangflowClient:
    """Async client for direct communication with the Langflow API."""

    def __init__(self):
        self.base_url = settings.langflow_url.rstrip("/")
        self.api_key = settings.langflow_api_key
        self.flow_id = settings.langflow_flow_id
        self.client = httpx.AsyncClient(
            timeout=120.0,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )

    async def close(self):
        await self.client.aclose()

    async def run_flow(
        self,
        message: str,
        session_id: Optional[str] = None,
        tweaks: Optional[dict] = None,
        stream: bool = True,
        flow_id: Optional[str] = None,
    ) -> dict:
        """
        Run a Langflow flow (non-streaming). Returns the full response.
        """
        target_flow = flow_id or self.flow_id
        url = f"{self.base_url}/api/v1/run/{target_flow}"

        payload = {
            "input_value": message,
            "output_type": "chat",
            "input_type": "chat",
        }
        if session_id:
            payload["session_id"] = session_id
        if tweaks:
            payload["tweaks"] = tweaks

        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract the text from Langflow's response structure
        try:
            outputs = data["outputs"][0]["outputs"][0]
            text = outputs["results"]["message"]["text"]
            return {"response": text, "session_id": session_id}
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Langflow response: {e}")
            return {"response": "", "session_id": session_id}

    async def stream_flow(
        self,
        message: str,
        session_id: Optional[str] = None,
        tweaks: Optional[dict] = None,
        flow_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a Langflow flow. Yields SSE-formatted lines compatible with
        the frontend's expected format:
            data: {"type": "content", "delta": "..."}
            data: {"type": "done", "chat_id": "..."}
        
        Langflow streams line-delimited JSON with events:
            {"event": "token", "data": {"chunk": "..."}}
            {"event": "add_message", "data": {...}}
            {"event": "end", "data": {"result": {...}}}
        """
        target_flow = flow_id or self.flow_id
        url = f"{self.base_url}/api/v1/run/{target_flow}?stream=true"

        payload = {
            "input_value": message,
            "output_type": "chat",
            "input_type": "chat",
        }
        if session_id:
            payload["session_id"] = session_id
        if tweaks:
            payload["tweaks"] = tweaks

        async with self.client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                logger.error(f"Langflow stream error {response.status_code}: {error_body.decode()}")
                yield f"data: {json.dumps({'error': f'Langflow error {response.status_code}'})}\n\n"
                return

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event_type = event.get("event")

                    if event_type == "token":
                        chunk = event.get("data", {}).get("chunk", "")
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"

                    elif event_type == "end":
                        yield f"data: {json.dumps({'type': 'done', 'chat_id': session_id})}\n\n"

                    # "add_message" events are metadata; skip them for streaming
                except json.JSONDecodeError:
                    logger.debug(f"Skipping non-JSON line from Langflow: {line[:100]}")


_langflow_client_instance = None


def get_langflow_client() -> LangflowClient:
    global _langflow_client_instance
    if _langflow_client_instance is None:
        _langflow_client_instance = LangflowClient()
    return _langflow_client_instance
