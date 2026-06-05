import json
import os
from typing import List, Optional
from app.models.persona import Persona, PersonaUpdate

# Refreshed persona service logic to load the new 6 brand personas.

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PERSONAS_FILE = os.path.join(DATA_DIR, "personas.json")

class PersonaService:
    def __init__(self):
        self.personas = self._load_personas()

    def _load_personas(self) -> List[Persona]:
        if not os.path.exists(PERSONAS_FILE):
            return []
        with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [Persona(**p) for p in data]

    def _save_personas(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(PERSONAS_FILE, "w", encoding="utf-8") as f:
            json.dump([p.model_dump() for p in self.personas], f, indent=2)

    def list_personas(self) -> List[Persona]:
        return self.personas

    def get_persona(self, persona_id: str) -> Optional[Persona]:
        for p in self.personas:
            if p.id == persona_id:
                return p
        return None

    def update_persona(self, persona_id: str, data: PersonaUpdate) -> Optional[Persona]:
        persona = self.get_persona(persona_id)
        if not persona:
            return None
        
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(persona, key, value)
        
        self._save_personas()
        return persona

_persona_service_instance = None

def get_persona_service() -> PersonaService:
    global _persona_service_instance
    if _persona_service_instance is None:
        _persona_service_instance = PersonaService()
    return _persona_service_instance
