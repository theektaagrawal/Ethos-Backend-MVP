from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.services.openrag_client import get_openrag_client, OpenRAGClient

router = APIRouter(prefix="/api/connectors", tags=["connectors"])

@router.get("")
@router.get("/")
async def list_connectors(client: OpenRAGClient = Depends(get_openrag_client)):
    response = await client.client.get("/connectors")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="OpenRAG error")
    return response.json()

@router.get("/{connector_type}/status")
async def get_connector_status(connector_type: str, client: OpenRAGClient = Depends(get_openrag_client)):
    response = await client.client.get(f"/connectors/{connector_type}/status")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="OpenRAG error")
    return response.json()

class SyncRequest(BaseModel):
    settings: Optional[Dict[str, Any]] = None

@router.post("/{connector_type}/sync")
async def sync_connector(connector_type: str, request: SyncRequest, client: OpenRAGClient = Depends(get_openrag_client)):
    payload = request.model_dump(exclude_unset=True) if request else {}
    response = await client.client.post(f"/connectors/{connector_type}/sync", json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"OpenRAG error: {response.text}")
    return response.json()
