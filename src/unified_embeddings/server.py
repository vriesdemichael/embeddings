from typing import Sequence
from fastapi import FastAPI
from unified_embeddings.model_router import create_model_router


from unified_embeddings.models.simple import SimpleModel

import logging

from unified_embeddings.servable_embedder import ServableEmbedder

logger = logging.getLogger("uvicorn.error")


def app_factory(models: Sequence[ServableEmbedder]) -> FastAPI:
    app = FastAPI(
        title="Unified Embeddings",
        description="Provides a unified interface for various embdding models",
    )

    for model in models:
        model_name = model.HF_NAME.split("/")[-1]

        logger.info("Mounting model: %s at %s", model.HF_NAME, model_name)
        app.include_router(
            create_model_router(model), prefix=f"/{model_name}", tags=[model_name]
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "OK"}

    return app


if __name__ == "__main__":
    models = [
        SimpleModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ]
    app = app_factory(models)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
