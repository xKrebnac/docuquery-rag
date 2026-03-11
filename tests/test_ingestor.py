"""Tests für src/ingestor.py – Phase 1: Dokument-Ingestion und Chunking."""

import io
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tiktoken

from src.ingestor import DocumentIngestor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ingestor() -> DocumentIngestor:
    return DocumentIngestor()


@pytest.fixture()
def enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture()
def tmp_txt(tmp_path: Path) -> Path:
    """Eine kleine TXT-Datei mit bekanntem Inhalt."""
    p = tmp_path / "sample.txt"
    p.write_text("Hello world. This is a test document.", encoding="utf-8")
    return p


@pytest.fixture()
def long_txt(tmp_path: Path, enc: tiktoken.Encoding) -> Path:
    """Eine TXT-Datei, deren Inhalt garantiert mehr als 512 Tokens umfasst."""
    # ~600 Tokens: ein bekanntes Wort wird vielfach wiederholt
    word = "token "
    tokens_needed = 600
    content = word * tokens_needed
    p = tmp_path / "long.txt"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_document – TXT
# ---------------------------------------------------------------------------

class TestLoadDocumentTxt:
    def test_returns_string(self, ingestor: DocumentIngestor, tmp_txt: Path) -> None:
        result = ingestor.load_document(tmp_txt)
        assert isinstance(result, str)

    def test_content_matches(self, ingestor: DocumentIngestor, tmp_txt: Path) -> None:
        result = ingestor.load_document(tmp_txt)
        assert "Hello world" in result

    def test_file_not_found(self, ingestor: DocumentIngestor) -> None:
        with pytest.raises(FileNotFoundError):
            ingestor.load_document("/nonexistent/path/doc.txt")

    def test_unsupported_extension(self, ingestor: DocumentIngestor, tmp_path: Path) -> None:
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingestor.load_document(f)


# ---------------------------------------------------------------------------
# load_document – PDF (gemockt)
# ---------------------------------------------------------------------------

class TestLoadDocumentPdf:
    def test_pdf_text_extracted(self, ingestor: DocumentIngestor, tmp_path: Path) -> None:
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")  # Inhalt irrelevant, PdfReader wird gemockt

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page one content."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("src.ingestor.PdfReader", return_value=mock_reader):
            result = ingestor.load_document(pdf_path)

        assert "Page one content." in result

    def test_pdf_multiple_pages_joined(self, ingestor: DocumentIngestor, tmp_path: Path) -> None:
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "First page."
        pages[1].extract_text.return_value = "Second page."
        mock_reader = MagicMock()
        mock_reader.pages = pages

        with patch("src.ingestor.PdfReader", return_value=mock_reader):
            result = ingestor.load_document(pdf_path)

        assert "First page." in result
        assert "Second page." in result

    def test_pdf_none_page_handled(self, ingestor: DocumentIngestor, tmp_path: Path) -> None:
        """Seiten, die None von extract_text zurückgeben, dürfen keinen Fehler verursachen."""
        pdf_path = tmp_path / "bad_page.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        page = MagicMock()
        page.extract_text.return_value = None
        mock_reader = MagicMock()
        mock_reader.pages = [page]

        with patch("src.ingestor.PdfReader", return_value=mock_reader):
            result = ingestor.load_document(pdf_path)

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_text_returns_empty_list(self, ingestor: DocumentIngestor) -> None:
        assert ingestor.chunk_text("") == []

    def test_short_text_single_chunk(self, ingestor: DocumentIngestor) -> None:
        chunks = ingestor.chunk_text("Hello world", chunk_size=512, overlap=64)
        assert len(chunks) == 1

    def test_chunk_token_count_within_limit(
        self, ingestor: DocumentIngestor, long_txt: Path
    ) -> None:
        text = long_txt.read_text(encoding="utf-8")
        chunks = ingestor.chunk_text(text, chunk_size=512, overlap=64)
        for chunk in chunks:
            assert len(chunk) <= 512

    def test_multiple_chunks_produced(
        self, ingestor: DocumentIngestor, long_txt: Path
    ) -> None:
        text = long_txt.read_text(encoding="utf-8")
        chunks = ingestor.chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) > 1

    def test_overlap_shared_tokens(self, ingestor: DocumentIngestor, long_txt: Path) -> None:
        """Das Ende von Chunk N muss dem Anfang von Chunk N+1 entsprechen (bis zu overlap Tokens)."""
        text = long_txt.read_text(encoding="utf-8")
        overlap = 64
        chunks = ingestor.chunk_text(text, chunk_size=512, overlap=overlap)
        for i in range(len(chunks) - 1):
            assert chunks[i][-overlap:] == chunks[i + 1][:overlap]

    def test_invalid_overlap_raises(self, ingestor: DocumentIngestor) -> None:
        with pytest.raises(ValueError, match="overlap"):
            ingestor.chunk_text("some text", chunk_size=64, overlap=64)

    def test_returns_list_of_lists(self, ingestor: DocumentIngestor) -> None:
        chunks = ingestor.chunk_text("Hello world", chunk_size=512, overlap=64)
        assert isinstance(chunks, list)
        assert isinstance(chunks[0], list)


# ---------------------------------------------------------------------------
# ingest (Integration)
# ---------------------------------------------------------------------------

class TestIngest:
    def test_returns_list_of_dicts(self, ingestor: DocumentIngestor, tmp_txt: Path) -> None:
        records = ingestor.ingest(tmp_txt)
        assert isinstance(records, list)
        assert len(records) >= 1
        assert isinstance(records[0], dict)

    def test_dict_keys_present(self, ingestor: DocumentIngestor, tmp_txt: Path) -> None:
        record = ingestor.ingest(tmp_txt)[0]
        assert {"text", "chunk_index", "source", "token_count"} == set(record.keys())

    def test_source_is_filename(self, ingestor: DocumentIngestor, tmp_txt: Path) -> None:
        record = ingestor.ingest(tmp_txt)[0]
        assert record["source"] == tmp_txt.name

    def test_chunk_index_sequential(self, ingestor: DocumentIngestor, long_txt: Path) -> None:
        records = ingestor.ingest(long_txt)
        for expected, record in enumerate(records):
            assert record["chunk_index"] == expected

    def test_token_count_matches_text(
        self, ingestor: DocumentIngestor, enc: tiktoken.Encoding, tmp_txt: Path
    ) -> None:
        records = ingestor.ingest(tmp_txt)
        for record in records:
            assert record["token_count"] == len(enc.encode(record["text"]))

    def test_token_count_within_chunk_size(
        self, ingestor: DocumentIngestor, long_txt: Path
    ) -> None:
        chunk_size = 256
        records = ingestor.ingest(long_txt, chunk_size=chunk_size, overlap=32)
        for record in records:
            assert record["token_count"] <= chunk_size

    def test_text_is_string(self, ingestor: DocumentIngestor, tmp_txt: Path) -> None:
        for record in ingestor.ingest(tmp_txt):
            assert isinstance(record["text"], str)

    def test_concatenated_text_covers_original(
        self,
        ingestor: DocumentIngestor,
        enc: tiktoken.Encoding,
        tmp_txt: Path,
    ) -> None:
        """Jedes Token des Originaldokuments muss in mindestens einem Chunk vorkommen."""
        original_tokens = enc.encode(tmp_txt.read_text(encoding="utf-8"))
        records = ingestor.ingest(tmp_txt)
        chunk_tokens_flat = [t for r in records for t in enc.encode(r["text"])]
        # Alle Original-Tokens müssen vorhanden sein (Reihenfolge kann durch Overlap abweichen)
        assert set(original_tokens).issubset(set(chunk_tokens_flat))
