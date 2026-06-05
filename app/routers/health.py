from fastapi import APIRouter
from app.services.openrag_client import get_openrag_client

router = APIRouter(prefix="/api/health", tags=["health"])

@router.get("")
@router.get("/")
async def health_check():
    client = get_openrag_client()
    try:
        is_openrag_up = await client.check_health()
        return {
            "status": "ok",
            "service": "folio",
            "openrag_connected": is_openrag_up
        }
    except Exception as e:
        return {
            "status": "ok",
            "service": "folio",
            "openrag_connected": False,
            "error": str(e)
        }
