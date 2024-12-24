from numpy import ndarray
from sentence_transformers import SentenceTransformer

single_or_multiple_str: type = str | list[str]


class ServableEmbedder:
    HF_NAME = "sentinel"  # Must be overwritten by subclasses

    def __init__(self):
        if self.HF_NAME == "sentinel":
            raise ValueError("HF_NAME must be overwritten by subclasses")
        self.model = SentenceTransformer(
            self.HF_NAME,
            trust_remote_code=True,  # This is only the code that is supplied in the huggingface model repo. Not some remote code loaded at runtime.
        )

    def embed_document(self, document: str | list[str]) -> ndarray:
        raise NotImplementedError("This method must be implemented by subclasses")

    def embed_query(self, query: str | list[str]) -> ndarray:
        raise NotImplementedError("This method must be implemented by subclasses")

    def classify(self, documents: str | list[str]) -> ndarray:
        raise NotImplementedError("This method must be implemented by subclasses")

    def cluster(self, documents: list) -> ndarray:
        raise NotImplementedError("This method must be implemented by subclasses")

    @property
    def can_embed_documents(self):
        return self.embed_document.__func__ is not ServableEmbedder.embed_document

    @property
    def can_embed_queries(self):
        return self.embed_query.__func__ is not ServableEmbedder.embed_query

    @property
    def can_classify(self):
        return self.classify.__func__ is not ServableEmbedder.classify

    @property
    def can_cluster(self):
        return self.cluster.__func__ is not ServableEmbedder.cluster

    @property
    def can_rerank(self):
        return self.can_embed_documents and self.can_embed_queries

    @property
    def capabilities(self):
        return {
            "embed_documents": self.can_embed_documents,
            "embed_queries": self.can_embed_queries,
            "classify": self.can_classify,
            "cluster": self.can_cluster,
            "rerank": self.can_rerank,
        }
