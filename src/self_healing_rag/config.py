from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Self-Healing RAG"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = True

    model_config = SettingsConfigDict(
        env_prefix="SHR_",
        case_sensitive=False,
    )


settings = Settings()
