from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    video_path: str
    chunk_t: int = Field(validation_alias="TIME_OF_CHUNK")
    chunk_frames: int = Field(validation_alias="EXTRACTED_FRAMES_PER_CHUNK")
    overlap: int = Field(validation_alias="OVERLAP_T")
    api_end_point: str = Field(validation_alias="GATE_END_POINT")
    log_level: str = Field(validation_alias="LOG_LEVEL", examples=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"])
    log_file_path: str = Field(validation_alias="LOG_FILE_PATH")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
