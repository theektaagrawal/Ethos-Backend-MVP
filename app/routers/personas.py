from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.persona import Persona, PersonaUpdate
from app.services.persona_service import PersonaService, get_persona_service

router = APIRouter(prefix="/api/personas", tags=["personas"])

@router.get("/", response_model=List[Persona])
async def list_personas(service: PersonaService = Depends(get_persona_service)):
    return service.list_personas()

@router.get("/{persona_id}", response_model=Persona)
async def get_persona(persona_id: str, service: PersonaService = Depends(get_persona_service)):
    persona = service.get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona

@router.put("/{persona_id}", response_model=Persona)
async def update_persona(persona_id: str, data: PersonaUpdate, service: PersonaService = Depends(get_persona_service)):
    persona = service.update_persona(persona_id, data)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona
