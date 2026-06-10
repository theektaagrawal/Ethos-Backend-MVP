from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openrag_url: str = "http://localhost:8000"
    openrag_api_key: str = ""
    folio_api_key: str = "dev-api-key"
    openai_api_key: str = ""
    langflow_url: str = "https://langflow.ayan-khan.com"
    langflow_api_key: str = ""
    langflow_flow_id: str = ""
    port: int = 8001
    
    # New Auth Settings
    google_client_id: str = ""
    google_client_secret: str = ""
    jwt_secret: str = "folio-super-secret-key-123"

    # OpenAI Models
    openai_chat_model: str = "gpt-5.4-mini"
    openai_image_model: str = "gpt-image-2"
    openai_audio_model: str = "whisper-1"

    # Debugging
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

