from numpy import ndarray
from sentence_transformers import SentenceTransformer
from unified_embeddings.servable_embedder import ServableEmbedder


class SimpleModel(ServableEmbedder):
    """
    This class can be used to provide embeddings without any model specific logic.
    It expects symetric retrieval of embeddings. And will not do classification, clustering or reranking.
    """

    def __init__(self, model_name: str):
        self.HF_NAME = model_name
        self.model = SentenceTransformer(model_name)

    def embed_document(self, document: str | list[str]) -> ndarray:
        return self.model.encode(document)
