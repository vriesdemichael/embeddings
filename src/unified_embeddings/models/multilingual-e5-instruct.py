from numpy import ndarray
from unified_embeddings.servable_embedder import ServableEmbedder


class MultiLingualE5Instruct(ServableEmbedder):
    HF_NAME = "intfloat/multilingual-e5-large-instruct"

    def embed_document(self, documents: str | list[str]) -> ndarray:
        return self.model.encode(documents)

    def embed_query(self, query: str | list[str], instruction: str | None) -> ndarray:
        if instruction is None:
            raise ValueError("This model requires an instruction to embed queries")

        return self.model.encode(query, prompt=f"Instruct: {instruction}\nQuery: ")
