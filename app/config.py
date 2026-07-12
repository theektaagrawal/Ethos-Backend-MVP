from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openrag_url: str = "http://localhost:8000"
    lightrag_url: str = "http://localhost:9621"
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
    # Mainline model that hosts the Responses API image_generation tool for
    # multi-turn, context-preserving draft edits. Must be a model on OpenAI's
    # image_generation tool support list (the tool picks the GPT Image model).
    # NOTE: gpt-5-mini is NOT on that list; gpt-5.4-mini IS, so we use it here
    # for image edits even if chat/audit run on gpt-5-mini elsewhere.
    openai_responses_model: str = "gpt-5.4-mini"
    # Image tool quality. OpenAI's prompting guide says small text and multi-font
    # layouts (our ads: feature bullets, price, spec blocks) need medium or high;
    # "auto" may pick lower. High costs ~4x medium per image — tune via env if
    # spend matters more than text fidelity.
    openai_image_quality: str = "high"

    # Debugging
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

