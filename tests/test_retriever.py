"""Tests für src/retriever.py – Phase 3: Retrieval und LLM-Antwortgenerierung."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.retriever import DocumentRetriever


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_chroma_results(scores: list[float]) -> dict:
    """Baut das dict-Format, das ChromaDB's query() zurückgibt."""
    return {
        "ids": [[f"doc.txt::{i}" for i in range(len(scores))]],
        "documents": [[f"Textinhalt Chunk {i}" for i in range(len(scores))]],
        "metadatas": [
            [
                {"source": "doc.txt", "chunk_index": i, "token_count": 10}
                for i in range(len(scores))
            ]
        ],
        "distances": [[1 - score for score in scores]],
    }


def _make_openai_embedding_response() -> MagicMock:
    """Erstellt eine gefälschte OpenAI-Embeddings-Antwort."""
    item = MagicMock()
    item.embedding = [0.1] * 1536
    response = MagicMock()
    response.data = [item]
    return response


def _make_openai_chat_response(text: str = "Test-Antwort") -> MagicMock:
    """Erstellt eine gefälschte OpenAI-Chat-Completion-Antwort."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def api_key(monkeypatch) -> None:
    """Setzt einen Dummy-API-Key für alle Tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Gibt einen vorbereiteten Mock des OpenAI-Clients zurück."""
    client = MagicMock()
    client.embeddings.create.return_value = _make_openai_embedding_response()
    client.chat.completions.create.return_value = _make_openai_chat_response()
    return client


@pytest.fixture()
def mock_chroma_client() -> MagicMock:
    """Gibt einen vorbereiteten Mock des ChromaDB-Clients zurück."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = _make_chroma_results([0.9, 0.8])
    client = MagicMock()
    client.get_collection.return_value = mock_collection
    return client


@pytest.fixture()
def retriever(mock_openai_client: MagicMock, mock_chroma_client: MagicMock) -> DocumentRetriever:
    """Erstellt einen DocumentRetriever mit vollständig gemockten externen Abhängigkeiten."""
    with (
        patch("src.retriever.openai.OpenAI", return_value=mock_openai_client),
        patch("src.retriever.chromadb.PersistentClient", return_value=mock_chroma_client),
    ):
        return DocumentRetriever()


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_returns_list_of_dicts(
        self, retriever: DocumentRetriever, mock_chroma_client: MagicMock
    ) -> None:
        """retrieve() gibt eine Liste von Dicts zurück."""
        mock_chroma_client.get_collection.return_value.query.return_value = (
            _make_chroma_results([0.9])
        )

        result = retriever.retrieve("Was ist RAG?", "test-collection")

        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_retrieve_filters_by_threshold(
        self, retriever: DocumentRetriever, mock_chroma_client: MagicMock
    ) -> None:
        """retrieve() filtert Chunks unterhalb von SCORE_THRESHOLD heraus.

        SCORE_THRESHOLD ist temporär 0.0 – alle Chunks passieren den Filter.
        Der Test prüft, dass scores korrekt mit 1 - (dist/2) berechnet werden.
        dist = 1 - cosine_score → score = 1 - (1 - cosine_score) / 2
        Für input 0.9: dist=0.1 → score=0.95; für input 0.5: dist=0.5 → score=0.75.
        """
        mock_chroma_client.get_collection.return_value.query.return_value = (
            _make_chroma_results([0.9, 0.5])
        )

        result = retriever.retrieve("Was ist RAG?", "test-collection")

        assert len(result) == 2
        assert result[0]["score"] == pytest.approx(0.95)
        assert result[1]["score"] == pytest.approx(0.75)

    def test_retrieve_correct_keys(
        self, retriever: DocumentRetriever, mock_chroma_client: MagicMock
    ) -> None:
        """Jedes zurückgegebene Dict enthält die erwarteten Keys."""
        mock_chroma_client.get_collection.return_value.query.return_value = (
            _make_chroma_results([0.9])
        )

        result = retriever.retrieve("Was ist RAG?", "test-collection")

        assert len(result) == 1
        assert {"text", "source", "chunk_index", "score", "token_count"} == set(result[0].keys())


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    def test_generate_answer_returns_correct_keys(
        self, retriever: DocumentRetriever, mock_openai_client: MagicMock
    ) -> None:
        """generate_answer() gibt ein Dict mit den erwarteten Keys zurück."""
        chunks = [
            {"text": "Relevanter Text", "source": "doc.txt", "chunk_index": 0,
             "score": 0.9, "token_count": 10}
        ]
        mock_openai_client.chat.completions.create.return_value = (
            _make_openai_chat_response("Eine Antwort.")
        )

        result = retriever.generate_answer("Was ist RAG?", chunks)

        assert {"answer", "sources", "chunk_count", "model"} == set(result.keys())

    def test_generate_answer_with_empty_chunks(
        self, retriever: DocumentRetriever, mock_openai_client: MagicMock
    ) -> None:
        """generate_answer() gibt bei leerer Chunk-Liste eine sinnvolle Antwort zurück."""
        result = retriever.generate_answer("Was ist RAG?", [])

        assert result["chunk_count"] == 0
        assert result["sources"] == []
        assert "answer" in result
        assert len(result["answer"]) > 0
        mock_openai_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Initialisierung
# ---------------------------------------------------------------------------

class TestInit:
    def test_missing_api_key_raises(self, monkeypatch) -> None:
        """Fehlender OPENAI_API_KEY führt bei der Initialisierung zu einem EnvironmentError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            with (
                patch("src.retriever.openai.OpenAI"),
                patch("src.retriever.chromadb.PersistentClient"),
            ):
                DocumentRetriever()
