from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.generator_service import GeneratorService, get_generator_service

router = APIRouter(prefix="/api/generator", tags=["generator"])


class GenerateRequest(BaseModel):
    prompt: str


@router.post("/generate")
async def generate_image(
    request: GenerateRequest,
    service: GeneratorService = Depends(get_generator_service),
):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    try:
        return StreamingResponse(service.generate_image(request.prompt.strip()), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
