from unified_embeddings.models.simple import SimpleModel
from unified_embeddings.server import app_factory


def main():
    import uvicorn

    models = [
        # SimpleModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
        SimpleModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        # SnowflakeArcticEmbedV2(),
        # NomicEmbedTextV15(),
    ]

    app = app_factory(models)

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
