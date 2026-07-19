from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    video_path: str
    chunk_t: int = Field(validation_alias='TIME_OF_CHUNK')
    chunk_frames: int = Field(validation_alias='EXTRACTED_FRAMES_PER_CHUNK')
    overlap: int = Field(validation_alias='OVERLAP_T')

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()