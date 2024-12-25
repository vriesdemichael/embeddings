from numpy import ndarray
from unified_embeddings.servable_embedder import ServableEmbedder


class SnowflakeArcticEmbedV2(ServableEmbedder):
    HF_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"

    def embed_document(self, document: str | list | str) -> ndarray:
        return self.model.encode(document)

    def embed_query(self, query: str | list[str]) -> ndarray:
        return self.model.encode(query, prompt_name="query")

