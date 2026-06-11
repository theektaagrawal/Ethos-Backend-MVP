from pydantic import BaseModel
from typing import List, Optional

class DraftAuditResponse(BaseModel):
    improvements: List[str]
    rejections: List[str]

class DraftApplyRequest(BaseModel):
    image_base64: str
    description: str
    improvements: List[str]
    rejections: List[str]
    brand_name: Optional[str] = None

class DraftApplyResponse(BaseModel):
    image_base64: str
