from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = 'churn-prediction-api'
    app_env: str = 'dev'
    log_level: str = 'INFO'

    openai_api_key: str | None = Field(default=None, alias='OPENAI_API_KEY')
    openai_base_url: str | None = Field(default=None, alias='OPENAI_BASE_URL')
    openai_model: str = Field(default='text-embedding-3-small', alias='OPENAI_MODEL')

    model_bundle_path: Path = Field(default=Path('artifacts/model/model_bundle.joblib'), alias='MODEL_BUNDLE_PATH')

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
