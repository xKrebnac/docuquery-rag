"""Dokument-Ingestion und Chunking für die DocuQuery-RAG-Pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Lädt Dokumente und teilt sie in tokenbasierte, überlappende Chunks auf."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def load_document(self, filepath: str | Path) -> str:
        """Liest eine PDF- oder TXT-Datei ein und gibt deren Rohtext zurück.

        Args:
            filepath: Pfad zum Dokument. Unterstützte Endungen: .pdf, .txt.

        Returns:
            Den vollständigen Textinhalt des Dokuments als einzelnen String.

        Raises:
            ValueError: Wenn die Dateiendung nicht unterstützt wird.
            FileNotFoundError: Wenn die Datei nicht existiert.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dokument nicht gefunden: {path}")

        ext = path.suffix.lower()
        logger.info("Lade Dokument: %s", path.name)

        if ext == ".txt":
            text = path.read_text(encoding="utf-8")
        elif ext == ".pdf":
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
            logger.debug("Text aus %d PDF-Seite(n) extrahiert", len(reader.pages))
        else:
            raise ValueError(f"Unsupported file type '{ext}'. Use .pdf or .txt.")

        logger.info("%d Zeichen aus '%s' geladen", len(text), path.name)
        return text

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[list[int]]:
        """Teilt *text* in überlappende Token-Chunks auf.

        Args:
            text: Rohtext, der aufgeteilt werden soll.
            chunk_size: Maximale Anzahl an Tokens pro Chunk.
            overlap: Anzahl gemeinsamer Tokens zwischen aufeinanderfolgenden Chunks.

        Returns:
            Eine Liste von Token-ID-Listen, jede mit einer Länge <= chunk_size.

        Raises:
            ValueError: Wenn overlap >= chunk_size.
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})."
            )

        tokens = self._enc.encode(text)
        total_tokens = len(tokens)
        logger.debug("Text ergibt %d Tokens", total_tokens)

        if total_tokens == 0:
            return []

        stride = chunk_size - overlap
        chunks: list[list[int]] = []
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunks.append(tokens[start:end])
            if end == total_tokens:
                break
            start += stride

        logger.debug(
            "%d Chunk(s) erzeugt (size=%d, overlap=%d)", len(chunks), chunk_size, overlap
        )
        return chunks

    def ingest(
        self,
        filepath: str | Path,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[dict[str, Any]]:
        """Lädt ein Dokument, zerlegt es in Chunks und gibt strukturierte Datensätze zurück.

        Args:
            filepath: Pfad zum Dokument.
            chunk_size: Maximale Tokens pro Chunk.
            overlap: Token-Überlappung zwischen aufeinanderfolgenden Chunks.

        Returns:
            Eine Liste von Dicts, jedes enthält:
              - ``text``        – dekodierter Chunk-Text
              - ``chunk_index`` – nullbasierte Position dieses Chunks
              - ``source``      – Dateiname des Ursprungsdokuments
              - ``token_count`` – Anzahl der Tokens in diesem Chunk
        """
        path = Path(filepath)
        text = self.load_document(path)
        token_chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        records: list[dict[str, Any]] = []
        for idx, token_ids in enumerate(token_chunks):
            records.append(
                {
                    "text": self._enc.decode(token_ids),
                    "chunk_index": idx,
                    "source": path.name,
                    "token_count": len(token_ids),
                }
            )

        logger.info(
            "'%s' verarbeitet → %d Chunk(s) (chunk_size=%d, overlap=%d)",
            path.name,
            len(records),
            chunk_size,
            overlap,
        )
        return records
