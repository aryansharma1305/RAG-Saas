from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="rag-saas-local", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    llama_cloud_api_key: str = Field(default="", alias="LLAMA_CLOUD_API_KEY")
    use_llama_parse: bool = Field(default=False, alias="USE_LLAMA_PARSE")

    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    app_api_key: str = Field(default="", alias="APP_API_KEY")
    default_user_id: str = Field(default="local_user", alias="DEFAULT_USER_ID")

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    reranking_enabled: bool = Field(default=False, alias="RERANKING_ENABLED")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    default_model_key: str = Field(default="gemma", alias="DEFAULT_MODEL_KEY")
    qwen_model: str = Field(default="qwen3:8b", alias="QWEN_MODEL")
    gemma_model: str = Field(default="gemma3:4b", alias="GEMMA_MODEL")
    glm_model: str = Field(default="glm4:9b", alias="GLM_MODEL")

    app_database_url: str = Field(default="sqlite:///data/rag.db", alias="APP_DATABASE_URL")

    @property
    def local_models(self) -> dict[str, str]:
        return {
            "qwen": self.qwen_model,
            "gemma": self.gemma_model,
            "glm": self.glm_model,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
