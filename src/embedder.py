"""Embedding-Erstellung und ChromaDB-Speicherung für die DocuQuery-RAG-Pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
import openai

logger = logging.getLogger(__name__)

_PERSIST_DIR = Path(__file__).parent.parent / ".chromadb"
_EMBEDDING_MODEL = "text-embedding-ada-002"
_BATCH_SIZE = 100


class DocumentEmbedder:
    """Erstellt Embeddings für Dokument-Chunks und verwaltet deren Speicherung in ChromaDB."""

    def __init__(self, persist_dir: str | Path = _PERSIST_DIR) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Umgebungsvariable OPENAI_API_KEY ist nicht gesetzt."
            )
        self._client = openai.OpenAI(api_key=api_key)
        self._db = chromadb.PersistentClient(path=str(persist_dir))

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Erstellt Embedding-Vektoren für eine Liste von Chunk-Dicts.

        Die Chunks werden in Batches von maximal 100 Einträgen an die
        OpenAI Embeddings API (Modell: text-embedding-ada-002) gesendet.
        Jeder Chunk wird um den Key ``embedding`` (Liste von floats)
        erweitert. Die Eingabeliste wird in-place verändert und
        zusätzlich zurückgegeben.

        Args:
            chunks: Liste von Chunk-Dicts, wie sie ``DocumentIngestor.ingest()``
                    zurückgibt. Jedes Dict muss den Key ``text`` enthalten.

        Returns:
            Dieselbe Liste, wobei jedes Dict nun den Key ``embedding`` trägt.
        """
        total = len(chunks)
        logger.info("Starte Embedding-Erstellung für %d Chunk(s).", total)

        for batch_start in range(0, total, _BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            batch_end = batch_start + len(batch)
            logger.debug(
                "Verarbeite Batch %d–%d von %d …", batch_start + 1, batch_end, total
            )

            texts = [chunk["text"] for chunk in batch]
            response = self._client.embeddings.create(
                model=_EMBEDDING_MODEL,
                input=texts,
            )

            for chunk, item in zip(batch, response.data):
                chunk["embedding"] = item.embedding

        logger.info("Embedding-Erstellung abgeschlossen (%d Chunk(s)).", total)
        return chunks

    def store(self, chunks: list[dict[str, Any]], collection_name: str) -> None:
        """Speichert Chunks mit ihren Embeddings und Metadaten in ChromaDB.

        Legt die Collection an, falls sie noch nicht existiert, und
        fügt alle Chunks per Upsert ein. Als eindeutige ID wird
        ``<source>::<chunk_index>`` verwendet, sodass Duplikate
        automatisch überschrieben werden.

        Args:
            chunks: Liste von Chunk-Dicts, die den Key ``embedding``
                    enthalten müssen (nach Aufruf von ``embed_chunks``).
            collection_name: Name der ChromaDB-Collection.
        """
        logger.info(
            "Speichere %d Chunk(s) in Collection '%s' …", len(chunks), collection_name
        )

        collection = self._db.get_or_create_collection(collection_name)

        ids = [f"{c['source']}::{c['chunk_index']}" for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
            }
            for c in chunks
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "%d Chunk(s) erfolgreich in Collection '%s' gespeichert.",
            len(chunks),
            collection_name,
        )

    def load_collection(self, collection_name: str) -> chromadb.Collection:
        """Lädt eine bestehende ChromaDB-Collection und gibt sie zurück.

        Args:
            collection_name: Name der zu ladenden Collection.

        Returns:
            Das ChromaDB-Collection-Objekt.

        Raises:
            ValueError: Wenn die Collection nicht existiert.
        """
        logger.info("Lade Collection '%s' …", collection_name)
        try:
            collection = self._db.get_collection(collection_name)
        except Exception as exc:
            raise ValueError(
                f"Collection '{collection_name}' existiert nicht in der Datenbank."
            ) from exc

        logger.info("Collection '%s' erfolgreich geladen.", collection_name)
        return collection
