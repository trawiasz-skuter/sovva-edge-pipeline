from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    opensearch_host: str
    opensearch_port: int = 9200
    opensearch_user: str
    opensearch_password: str
    api_port: int = 8081

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
