from fastapi import APIRouter
from pydantic import BaseModel
from unified_embeddings.servable_embedder import ServableEmbedder


class TokenizerInformation(BaseModel):
    vocab_size: int
    special_tokens: dict
    padding_side: str
    truncation_side: str
    model_max_length: int


class Capabilities(BaseModel):
    embed_documents: bool
    embed_queries: bool
    classify: bool
    cluster: bool
    rerank: bool


class ModelInformation(BaseModel):
    model_name: str
    model_size: str
    output_dimensions: int
    max_sequence_length: int
    tokenizer_info: TokenizerInformation
    device: str
    n_model_params: int
    precision: str
    memory_usage: str
    capabilities: Capabilities


def create_model_router(model: ServableEmbedder) -> APIRouter:
    router = APIRouter()

    n_params = sum(p.numel() for p in model.model.parameters())
    n_params_human_readable = (
        f"{n_params / 1e6:.1f}M" if n_params < 1e9 else f"{n_params / 1e9:.1f}B"
    )
    precision = str(next(model.model.parameters()).dtype)
    dtype_to_bytes = {
        "torch.float32": 4,
        "torch.float": 4,
        "torch.float64": 8,
        "torch.double": 8,
        "torch.float16": 2,
        "torch.half": 2,
        "torch.bfloat16": 2,
        "torch.uint8": 1,
        "torch.int8": 1,
        "torch.int16": 2,
        "torch.short": 2,
        "torch.int32": 4,
        "torch.int": 4,
        "torch.int64": 8,
        "torch.long": 8,
        "torch.bool": 1,
    }

    if precision in dtype_to_bytes:
        memory_usage = n_params * dtype_to_bytes[precision]
        memory_usage_human_readable = (
            f"{memory_usage / 1e6:.1f}MB"
            if memory_usage < 1e9
            else f"{memory_usage / 1e9:.1f}GB"
        )

    tokenizer_info = TokenizerInformation(
        vocab_size=model.model.tokenizer.vocab_size,
        special_tokens=model.model.tokenizer.special_tokens_map_extended,
        padding_side=model.model.tokenizer.padding_side,
        truncation_side=model.model.tokenizer.truncation_side,
        model_max_length=model.model.tokenizer.model_max_length,
    )
    model_information = ModelInformation(
        model_name=model.HF_NAME,
        model_size=n_params_human_readable,
        output_dimensions=model.model.get_sentence_embedding_dimension(),
        max_sequence_length=model.model.get_max_seq_length(),
        tokenizer_info=tokenizer_info,
        device=str(model.model.device),
        n_model_params=n_params,
        precision=precision,
        memory_usage=memory_usage_human_readable,
        capabilities=model.capabilities,
    )

    if model.can_embed_documents:

        @router.post("/embed-documents")
        async def embed_document(
            documents: str | list[str],
        ) -> list[float] | list[list[float]]:
            """
            Create embeddings for a text or list of texts.
            You can use this endpoint for matching texts to similar texts or for creating a database of embeddings to be used for retrieval.
            """
            return model.embed_document(documents).tolist()

    if model.can_embed_queries:

        @router.post("/embed-queries")
        async def embed_query(
            queries: str | list[str],
        ) -> list[float] | list[list[float]]:
            """
            Create embeddings for a text or list of texts.
            You can use this endpoint to create embeddings for queries to be used for retrieval.
            Use this when you match a query to a set of documents. Or a question to answers.
            """
            return model.embed_query(queries).tolist()

    if model.can_classify:

        @router.post("/classify")
        async def classify(
            documents: str | list[str],
        ) -> list[float] | list[list[float]]:
            """
            Create embeddings for a text or list of texts.
            The embeddings from this endpoint are suited to be used for downstream classification tasks.
            """
            return model.classify(documents).tolist()

    if model.can_cluster:

        @router.post("/cluster")
        async def cluster(
            documents: str | list[str],
        ) -> list[float] | list[list[float]]:
            """
            Create embeddings for a text or list of texts.
            The embeddings from this endpoint are suited to be used for clustering tasks.
            """
            return model.cluster(documents).tolist()

    if model.can_rerank:

        @router.post("/rerank")
        async def rerank(
            documents: list[str], queries: str | list[str]
        ) -> list[float] | list[list[float]]:
            """
            Creates embeddings for documents and queries (assymtrically) and calculates the similarity between them.
            """
            document_embeddings = model.embed_document(documents)
            query_embeddings = model.embed_query(queries)

            return (query_embeddings @ document_embeddings.T).tolist()

    @router.get("/model_info", response_model=ModelInformation)
    async def model_info() -> ModelInformation:
        return model_information

    return router
