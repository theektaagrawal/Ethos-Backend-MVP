from pydantic import BaseModel
from typing import List, Optional

class DraftFinding(BaseModel):
    """One structured audit finding. An element gets exactly one verdict, so
    contradictory instructions (restyle X + remove X) cannot coexist."""
    element: str
    verdict: str  # "restyle" | "remove"
    violation: str
    fix: str
    source: Optional[str] = None

class DraftAuditResponse(BaseModel):
    improvements: List[str]
    rejections: List[str]
    compliant: Optional[bool] = None
    findings: Optional[List[DraftFinding]] = None
    preserve: Optional[List[str]] = None

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
    # Structured audit output. When present, the edit prompt is assembled
    # deterministically in code from these — no synthesis LLM call. The legacy
    # improvements list is only used as a fallback when findings are absent.
    findings: Optional[List[DraftFinding]] = None
    preserve: Optional[List[str]] = None

class DraftApplyResponse(BaseModel):
    image_base64: str
    # Responses API id for this edit turn; the client passes it back as
    # previous_response_id on the next apply to preserve context.
    response_id: Optional[str] = None
