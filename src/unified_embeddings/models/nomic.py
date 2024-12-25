from numpy import ndarray
from unified_embeddings.servable_embedder import ServableEmbedder


class NomicEmbedTextV15(ServableEmbedder):
    HF_NAME = "nomic-ai/nomic-embed-text-v1.5"

    def embed_document(self, document: str | list[str]) -> ndarray:
        return self.model.encode(document, prompt="search_document: ")

    def embed_query(self, query: str | list[str]) -> ndarray:
        return self.model.encode(query, prompt="search_qeury: " )

    def cluster(self, documents: str | list[str]) -> ndarray:
        return self.model.encode(documents, prompt="clustering: ")

    def classify(self, documents: str | list[str]) -> ndarray:
        return self.model.encode(documents, prompt="classification: ")

    def rerank(self, documents: list[str], queries: list[str]) -> ndarray:
        query_embeddings = self.embed_query(queries)
        document_embeddings = self.embed_document(documents)
        return query_embeddings @ document_embeddings.T
