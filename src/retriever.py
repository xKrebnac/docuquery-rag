"""Retrieval-Layer und LLM-Antwortgenerierung für die DocuQuery-RAG-Pipeline."""

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
_CHAT_MODEL = "gpt-3.5-turbo"


class DocumentRetriever:
    """Führt semantische Suche durch und generiert Antworten via GPT-3.5-turbo."""

    SCORE_THRESHOLD = 0.3

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

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Sucht semantisch ähnliche Chunks zur Abfrage in der angegebenen Collection.

        Args:
            query: Die Suchabfrage als Text.
            collection_name: Name der ChromaDB-Collection.
            top_k: Maximale Anzahl zurückgegebener Chunks.

        Returns:
            Gefilterte Liste von Chunk-Dicts mit text, source, chunk_index,
            score und token_count. Nur Chunks mit score >= SCORE_THRESHOLD
            werden zurückgegeben.
        """
        logger.info("Starte Retrieval für Abfrage: '%s'", query)

        response = self._client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=[query],
        )
        embedding = response.data[0].embedding

        collection = self._db.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict[str, Any]] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1 - (dist / 2)
            if score < self.SCORE_THRESHOLD:
                continue
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": score,
                    "token_count": meta.get("token_count", 0),
                }
            )

        logger.info("Retrieval abgeschlossen: %d relevante Chunk(s) gefunden.", len(chunks))
        return chunks

    def generate_answer(self, query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Generiert eine Antwort auf die Abfrage basierend auf den übergebenen Chunks.

        Args:
            query: Die ursprüngliche Nutzerfrage.
            chunks: Relevante Dokument-Chunks (Ausgabe von ``retrieve``).

        Returns:
            Dict mit answer, sources (unique), chunk_count und model.
        """
        if not chunks:
            logger.info("Keine relevanten Chunks vorhanden – überspringe API-Aufruf.")
            return {
                "answer": "Keine relevanten Dokumentenauszüge gefunden.",
                "sources": [],
                "chunk_count": 0,
                "model": _CHAT_MODEL,
            }

        context_parts = [
            f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(chunks)
        ]
        context = "\n\n".join(context_parts)

        system_message = (
            "Du bist ein hilfreicher Assistent. Beantworte die Frage ausschließlich "
            "auf Basis der folgenden Dokumentenauszüge. Falls die Auszüge keine "
            "ausreichenden Informationen enthalten, weise darauf hin."
        )
        user_message = f"Dokumentenauszüge:\n{context}\n\nFrage: {query}"

        logger.info("Generiere Antwort mit %s für %d Chunk(s).", _CHAT_MODEL, len(chunks))

        response = self._client.chat.completions.create(
            model=_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

        return {
            "answer": answer,
            "sources": sources,
            "chunk_count": len(chunks),
            "model": _CHAT_MODEL,
        }
