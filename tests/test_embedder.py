"""Tests für src/embedder.py – Phase 2: Embedding-Erstellung und ChromaDB-Speicherung."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.embedder import DocumentEmbedder


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 1) -> list[dict]:
    """Erstellt n minimale Chunk-Dicts im Format von DocumentIngestor.ingest()."""
    return [
        {
            "text": f"Testinhalt Chunk {i}",
            "chunk_index": i,
            "source": "dokument.txt",
            "token_count": 5,
        }
        for i in range(n)
    ]


def _make_openai_response(n: int = 1, dim: int = 1536) -> MagicMock:
    """Erstellt eine gefälschte OpenAI-Embeddings-Antwort mit n Vektoren der Dimension dim."""
    items = []
    for _ in range(n):
        item = MagicMock()
        item.embedding = [0.1] * dim
        items.append(item)
    response = MagicMock()
    response.data = items
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def api_key(monkeypatch) -> None:
    """Setzt einen Dummy-API-Key für alle Tests, damit kein echter Schlüssel nötig ist."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Gibt einen vorbereiteten Mock des OpenAI-Clients zurück."""
    client = MagicMock()
    client.embeddings.create.return_value = _make_openai_response(n=1)
    return client


@pytest.fixture()
def mock_chroma_client() -> MagicMock:
    """Gibt einen vorbereiteten Mock des ChromaDB-Clients zurück."""
    return MagicMock()


@pytest.fixture()
def embedder(mock_openai_client: MagicMock, mock_chroma_client: MagicMock) -> DocumentEmbedder:
    """Erstellt einen DocumentEmbedder mit vollständig gemockten externen Abhängigkeiten."""
    with (
        patch("src.embedder.openai.OpenAI", return_value=mock_openai_client),
        patch("src.embedder.chromadb.PersistentClient", return_value=mock_chroma_client),
    ):
        return DocumentEmbedder()


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------

class TestEmbedChunks:
    def test_embed_chunks_adds_embedding_key(
        self, embedder: DocumentEmbedder, mock_openai_client: MagicMock
    ) -> None:
        """Jeder Chunk enthält nach dem Aufruf den Key 'embedding'."""
        mock_openai_client.embeddings.create.return_value = _make_openai_response(n=1)
        chunks = _make_chunks(n=1)

        result = embedder.embed_chunks(chunks)

        assert "embedding" in result[0]

    def test_embed_chunks_correct_dimension(
        self, embedder: DocumentEmbedder, mock_openai_client: MagicMock
    ) -> None:
        """Der Embedding-Vektor hat die erwartete Dimension von 1536."""
        mock_openai_client.embeddings.create.return_value = _make_openai_response(n=1, dim=1536)
        chunks = _make_chunks(n=1)

        result = embedder.embed_chunks(chunks)

        assert len(result[0]["embedding"]) == 1536

    def test_embed_chunks_returns_same_list(
        self, embedder: DocumentEmbedder, mock_openai_client: MagicMock
    ) -> None:
        """embed_chunks gibt dieselbe (erweiterte) Liste zurück."""
        mock_openai_client.embeddings.create.return_value = _make_openai_response(n=3)
        chunks = _make_chunks(n=3)

        result = embedder.embed_chunks(chunks)

        assert result is chunks

    def test_embed_chunks_batching(
        self, embedder: DocumentEmbedder, mock_openai_client: MagicMock
    ) -> None:
        """Bei 150 Chunks werden genau 2 API-Aufrufe gemacht (Batch-Größe 100)."""
        mock_openai_client.embeddings.create.side_effect = [
            _make_openai_response(n=100),
            _make_openai_response(n=50),
        ]
        chunks = _make_chunks(n=150)

        embedder.embed_chunks(chunks)

        assert mock_openai_client.embeddings.create.call_count == 2

    def test_embed_chunks_all_chunks_get_embedding(
        self, embedder: DocumentEmbedder, mock_openai_client: MagicMock
    ) -> None:
        """Alle Chunks – auch über Batch-Grenzen hinweg – erhalten einen Embedding-Vektor."""
        mock_openai_client.embeddings.create.side_effect = [
            _make_openai_response(n=100),
            _make_openai_response(n=50),
        ]
        chunks = _make_chunks(n=150)

        result = embedder.embed_chunks(chunks)

        assert all("embedding" in c for c in result)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class TestStore:
    def test_store_creates_collection(
        self, embedder: DocumentEmbedder, mock_chroma_client: MagicMock
    ) -> None:
        """store() legt die angegebene Collection in ChromaDB an (oder lädt sie)."""
        chunks = _make_chunks(n=2)
        for c in chunks:
            c["embedding"] = [0.1] * 1536

        embedder.store(chunks, "test-collection")

        mock_chroma_client.get_or_create_collection.assert_called_once_with("test-collection")

    def test_store_calls_upsert(
        self, embedder: DocumentEmbedder, mock_chroma_client: MagicMock
    ) -> None:
        """store() ruft upsert() auf der Collection auf."""
        mock_collection = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        chunks = _make_chunks(n=1)
        chunks[0]["embedding"] = [0.1] * 1536

        embedder.store(chunks, "test-collection")

        mock_collection.upsert.assert_called_once()

    def test_store_ids_are_unique_per_source_and_index(
        self, embedder: DocumentEmbedder, mock_chroma_client: MagicMock
    ) -> None:
        """Die IDs folgen dem Schema '<source>::<chunk_index>'."""
        mock_collection = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        chunks = _make_chunks(n=2)
        for c in chunks:
            c["embedding"] = [0.1] * 1536

        embedder.store(chunks, "test-collection")

        _, kwargs = mock_collection.upsert.call_args
        assert kwargs["ids"] == ["dokument.txt::0", "dokument.txt::1"]

    def test_store_metadata_keys(
        self, embedder: DocumentEmbedder, mock_chroma_client: MagicMock
    ) -> None:
        """Jede Metadaten-Einheit enthält source, chunk_index und token_count."""
        mock_collection = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        chunks = _make_chunks(n=1)
        chunks[0]["embedding"] = [0.1] * 1536

        embedder.store(chunks, "test-collection")

        _, kwargs = mock_collection.upsert.call_args
        meta = kwargs["metadatas"][0]
        assert {"source", "chunk_index", "token_count"} == set(meta.keys())


# ---------------------------------------------------------------------------
# load_collection
# ---------------------------------------------------------------------------

class TestLoadCollection:
    def test_load_collection_returns_collection(
        self, embedder: DocumentEmbedder, mock_chroma_client: MagicMock
    ) -> None:
        """load_collection() gibt das Collection-Objekt zurück."""
        mock_collection = MagicMock()
        mock_chroma_client.get_collection.return_value = mock_collection

        result = embedder.load_collection("vorhandene-collection")

        assert result is mock_collection

    def test_load_collection_raises_on_missing(
        self, embedder: DocumentEmbedder, mock_chroma_client: MagicMock
    ) -> None:
        """load_collection() wirft ValueError mit deutschem Hinweis bei fehlender Collection."""
        mock_chroma_client.get_collection.side_effect = ValueError("Collection not found")

        with pytest.raises(ValueError, match="existiert nicht"):
            embedder.load_collection("nicht-vorhanden")


# ---------------------------------------------------------------------------
# Initialisierung
# ---------------------------------------------------------------------------

class TestInit:
    def test_missing_api_key_raises(self, monkeypatch) -> None:
        """Fehlender OPENAI_API_KEY führt bei der Initialisierung zu einem EnvironmentError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            with (
                patch("src.embedder.openai.OpenAI"),
                patch("src.embedder.chromadb.PersistentClient"),
            ):
                DocumentEmbedder()
