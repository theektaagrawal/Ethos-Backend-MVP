from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from app.services.validator_service import ValidatorService, get_validator_service
from app.models.validator import DraftApplyRequest
import base64

router = APIRouter(prefix="/api/validator", tags=["validator"])

@router.post("/audit")
async def audit_draft(
    description: str = Form(""),
    image: UploadFile = File(...),
    brand_name: str = Form("McKINLEY"),
    service: ValidatorService = Depends(get_validator_service)
):
    try:
        contents = await image.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        return StreamingResponse(service.audit_image_draft(image_base64, description, brand_name), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply")
async def apply_improvements(
    request: DraftApplyRequest,
    service: ValidatorService = Depends(get_validator_service)
):
    try:
        return StreamingResponse(service.apply_image_improvements(
            image_base64=request.image_base64,
            description=request.description,
            improvements=request.improvements,
            rejections=request.rejections,
            brand_name=request.brand_name or "McKINLEY"
        ), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
