from pydantic import BaseModel
from typing import Optional

class Persona(BaseModel):
    id: str
    name: str
    role: str
    flow_id: Optional[str] = None
    system_prompt: str
    icon: str
    color: str
    description: str

class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    flow_id: Optional[str] = None
    system_prompt: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
