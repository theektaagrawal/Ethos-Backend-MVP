import httpx
from fastapi import HTTPException
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class OpenRAGClient:
    def __init__(self):
        self.base_url = settings.openrag_url.rstrip("/")
        headers = {}
        if settings.openrag_api_key:
            # Depending on OpenRAG's exact auth method, this is usually Bearer or X-API-Key
            headers["Authorization"] = f"Bearer {settings.openrag_api_key}"
            headers["x-api-key"] = settings.openrag_api_key
            
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0, headers=headers)

    async def close(self):
        await self.client.aclose()

    async def check_health(self):
        try:
            response = await self.client.get("/api/v1/health") # Assuming a standard health endpoint, fallback to root if 404
            if response.status_code == 404:
                response = await self.client.get("/")
            return response.status_code in (200, 404, 403, 401) # If we get a response, it's alive
        except Exception as e:
            logger.warning(f"OpenRAG health check failed: {e}")
            return False

_client_instance = None

def get_openrag_client() -> OpenRAGClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenRAGClient()
    return _client_instance
