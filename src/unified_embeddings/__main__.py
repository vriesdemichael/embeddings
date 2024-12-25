from unified_embeddings.models.nomic import NomicEmbedTextV15
from unified_embeddings.models.simple import SimpleModel
from unified_embeddings.models.snowflake_arctic_embed_v2 import SnowflakeArcticEmbedV2
from unified_embeddings.server import app_factory


def main():
    import uvicorn

    models = [
        # SimpleModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
        SimpleModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        SnowflakeArcticEmbedV2(),
        NomicEmbedTextV15(),
    ]

    app = app_factory(models)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
