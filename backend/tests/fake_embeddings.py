class FakeEmbeddings:
    """Deterministic vectors of a fixed size; counts calls."""

    def __init__(self, dim=8):
        self.dim, self.calls = dim, 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(t) % 7)] + [0.5] * (self.dim - 1) for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FailingEmbeddings:
    def embed_documents(self, texts):
        raise ConnectionError("provider down")


class FixedEmbeddings:
    """Returns one chosen vector for any query."""

    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, text):
        return self.vector
