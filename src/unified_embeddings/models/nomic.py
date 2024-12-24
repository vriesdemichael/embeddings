from numpy import ndarray
from unified_embeddings.servable_embedder import ServableEmbedder


class NomicEmbedTextV15(ServableEmbedder):
    HF_NAME = "nomic-ai/nomic-embed-text-v1.5"

    def _prefix_with_task(self, texts: str | list[str], task: str) -> str | list[str]:
        if isinstance(texts, str):
            return f"{task}: {texts}"
        return [f"{task}: {text}" for text in texts]

    def embed_document(self, document: str | list[str]) -> ndarray:
        prepared_documents = self._prefix_with_task(document, "search_document")
        return self.model.encode(prepared_documents)

    def embed_query(self, query: str | list[str]) -> ndarray:
        prepared_queries = self._prefix_with_task(query, "search_query")
        return self.model.encode(prepared_queries)

    def cluster(self, documents: str | list[str]) -> ndarray:
        prepared_documents = self._prefix_with_task(documents, "clustering")
        return self.model.encode(prepared_documents)

    def classify(self, documents: str | list[str]) -> ndarray:
        prepared_documents = self._prefix_with_task(documents, "classification")
        return self.model.encode(prepared_documents)

    def rerank(self, documents: list[str], queries: list[str]) -> ndarray:
        query_embeddings = self.embed_query(queries)
        document_embeddings = self.embed_document(documents)
        return query_embeddings @ document_embeddings.T
