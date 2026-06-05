import jwt
import datetime
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from fastapi import HTTPException
from app.models.user import User
from sqlalchemy.orm import Session

# Note: Secret key should be properly loaded from env variables in production
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-me")
ALGORITHM = "HS256"

# Used to verify the Google token
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

def verify_google_token(token: str) -> dict:
    try:
        # Verify the token with Google
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        
        # Returns the decoded user information
        return {
            "google_id": idinfo['sub'],
            "email": idinfo['email'],
            "name": idinfo.get('name', ''),
            "avatar_url": idinfo.get('picture', '')
        }
    except ValueError:
        # Invalid token
        raise HTTPException(status_code=401, detail="Invalid Google token")

def create_access_token(user_id: int) -> str:
    expiration = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    payload = {
        "sub": str(user_id),
        "exp": expiration
    }
    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_or_create_user(db: Session, google_info: dict) -> User:
    user = db.query(User).filter(User.google_id == google_info["google_id"]).first()
    if not user:
        user = User(
            google_id=google_info["google_id"],
            email=google_info["email"],
            name=google_info["name"],
            avatar_url=google_info["avatar_url"]
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
