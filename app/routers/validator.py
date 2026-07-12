import json

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
    # JSON array of fix strings applied in the previous round. Iteration memory:
    # keeps the audit from re-flagging elements the last round already corrected.
    previous_fixes: str = Form(""),
    service: ValidatorService = Depends(get_validator_service)
):
    try:
        contents = await image.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        mime = image.content_type or "image/jpeg"

        fixes: list[str] = []
        if previous_fixes:
            try:
                parsed = json.loads(previous_fixes)
                if isinstance(parsed, list):
                    fixes = [str(f) for f in parsed if str(f).strip()]
            except json.JSONDecodeError:
                pass

        return StreamingResponse(
            service.audit_image_draft(
                f"data:{mime};base64,{image_base64}",
                description,
                brand_name,
                previous_fixes=fixes,
            ),
            media_type="text/event-stream",
        )
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
            brand_name=request.brand_name or "McKINLEY",
            previous_response_id=request.previous_response_id,
            findings=[f.model_dump() for f in request.findings] if request.findings else None,
            preserve=request.preserve,
        ), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
