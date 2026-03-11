"""CLI-Einstiegspunkt für die DocuQuery-RAG-Pipeline.

Verwendung:
    python -m src.cli ingest --file <pfad> --collection <name>
    python -m src.cli query  --question <frage> --collection <name> [--top-k N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Subkommandos
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingestion: Datei einlesen, chunken, embedden und in ChromaDB speichern."""
    from src.embedder import DocumentEmbedder
    from src.ingestor import DocumentIngestor

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Fehler: Datei nicht gefunden: {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Lade und chunke Dokument: {filepath.name} …")
    ingestor = DocumentIngestor()
    chunks = ingestor.ingest(filepath)
    print(f"      {len(chunks)} Chunk(s) erstellt.")

    print("[2/3] Erstelle Embeddings …")
    embedder = DocumentEmbedder()
    embedder.embed_chunks(chunks)
    print(f"      {len(chunks)} Embedding(s) generiert.")

    print(f"[3/3] Speichere in Collection '{args.collection}' …")
    embedder.store(chunks, args.collection)
    print(f"      Fertig. {len(chunks)} Chunk(s) gespeichert.")


def cmd_query(args: argparse.Namespace) -> None:
    """Query: Semantisch suchen und Antwort generieren."""
    from src.retriever import DocumentRetriever

    retriever = DocumentRetriever()

    print(f"Suche in Collection '{args.collection}' (top_k={args.top_k}) …")
    chunks = retriever.retrieve(args.question, args.collection, top_k=args.top_k)
    print(f"{len(chunks)} relevante(r) Chunk(s) gefunden.\n")

    result = retriever.generate_answer(args.question, chunks)

    print("=" * 60)
    print("Antwort:")
    print(result["answer"])
    print()
    if result["sources"]:
        print("Quellen:")
        for src in result["sources"]:
            print(f"  - {src}")
    else:
        print("Quellen: keine")
    print(f"\nVerwendete Chunks: {result['chunk_count']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Argument-Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docuquery",
        description="DocuQuery RAG – Dokumente einlesen und per LLM befragen.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="BEFEHL")
    subparsers.required = True

    # -- ingest ---------------------------------------------------------------
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Dokument einlesen, chunken und in ChromaDB speichern.",
    )
    p_ingest.add_argument("--file", required=True, metavar="PFAD",
                          help="Pfad zur .txt- oder .pdf-Datei.")
    p_ingest.add_argument("--collection", default="default", metavar="NAME",
                          help="Name der ChromaDB-Collection (Standard: default).")
    p_ingest.set_defaults(func=cmd_ingest)

    # -- query ----------------------------------------------------------------
    p_query = subparsers.add_parser(
        "query",
        help="Frage stellen und Antwort aus dem Dokumentbestand generieren.",
    )
    p_query.add_argument("--question", required=True, metavar="FRAGE",
                         help="Die zu beantwortende Frage.")
    p_query.add_argument("--collection", default="default", metavar="NAME",
                         help="Name der ChromaDB-Collection (Standard: default).")
    p_query.add_argument("--top-k", type=int, default=5, metavar="N",
                         help="Maximale Anzahl zurückgegebener Chunks (Standard: 5).")
    p_query.set_defaults(func=cmd_query)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
