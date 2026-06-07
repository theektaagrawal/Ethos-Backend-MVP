from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, personas, chat, knowledge, upload, connectors, auth, validator
from app.database import engine
from app.models.user import Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Ethos Folio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(personas.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(upload.router)
app.include_router(connectors.router)
app.include_router(validator.router)

@app.on_event("startup")
async def startup_event():
    # Warm up client or check connection
    pass

@app.on_event("shutdown")
async def shutdown_event():
    from app.services.openrag_client import get_openrag_client
    client = get_openrag_client()
    await client.close()
