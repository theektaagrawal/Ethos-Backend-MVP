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
    # When present, the edit is threaded onto a prior Responses API turn so the
    # model refines its own previous image at high fidelity instead of
    # re-ingesting a flattened re-upload. Set by the client after the first apply.
    previous_response_id: Optional[str] = None

class DraftApplyResponse(BaseModel):
    image_base64: str
    # Responses API id for this edit turn; the client passes it back as
    # previous_response_id on the next apply to preserve context.
    response_id: Optional[str] = None
