from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    app_name: str = "Self-Healing RAG"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = True

    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimension: int = 1024
    generation_model: str = "qwen2.5:3b"

    # Provisional retrieval-evaluation threshold.
    # 0.4 passed the current 45-query evaluation, but has not
    # been established as optimal for a broader production workload.
    retrieval_min_relevance_score: float = Field(
        default=0.4,
        ge=-1.0,
        le=1.0,
    )

    postgres_user: str = Field(
        validation_alias="POSTGRES_USER",
    )
    postgres_password: str = Field(
        validation_alias="POSTGRES_PASSWORD",
    )
    postgres_db: str = Field(
        validation_alias="POSTGRES_DB",
    )
    postgres_host: str = Field(
        default="localhost",
        validation_alias="POSTGRES_HOST",
    )
    postgres_port: int = Field(
        default=5432,
        validation_alias="POSTGRES_PORT",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SHR_",
        case_sensitive=False,
    )

    @property
    def database_url(self) -> URL:
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


settings = Settings()