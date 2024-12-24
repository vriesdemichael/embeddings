from unified_embeddings.servable_embedder import ServableEmbedder


class ImplementedEmbedder(ServableEmbedder):
    def __init__(self):
        self.HF_NAME = "test-implementation"
        # prevents the model from actually loading

    def embed_query(self, query):
        return None


def test_capabilities():
    impl = ImplementedEmbedder()
    assert impl.can_embed_queries
    assert not impl.can_embed_documents
    assert not impl.can_classify
    assert not impl.can_cluster
    assert not impl.can_rerank
