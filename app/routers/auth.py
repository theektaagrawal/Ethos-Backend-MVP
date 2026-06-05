from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.auth import verify_google_token, get_or_create_user, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

class GoogleLoginRequest(BaseModel):
    token: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

@router.post("/google", response_model=LoginResponse)
def google_login(request: GoogleLoginRequest, db: Session = Depends(get_db)):
    # 1. Verify token with Google
    google_info = verify_google_token(request.token)
    
    # 2. Find or create user in our DB
    user = get_or_create_user(db, google_info)
    
    # 3. Generate our own JWT session token
    access_token = create_access_token(user.id)
    
    return LoginResponse(
        access_token=access_token,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "avatar_url": user.avatar_url
        }
    )
