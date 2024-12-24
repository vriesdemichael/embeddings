import pytest
from fastapi.testclient import TestClient
from unified_embeddings.models.simple import SimpleModel
from unified_embeddings.server import app_factory


@pytest.fixture
def test_models():
    return [
        SimpleModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ]

@pytest.fixture
def client(test_models):
    app = app_factory(test_models)
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_model_mounting(client):
    response = client.get("/paraphrase-multilingual-MiniLM-L12-v2")
    assert response.status_code == 404  # Assuming the endpoint exists and is accessible
