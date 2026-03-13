# DocuQuery RAG

DocuQuery ist eine lokale Retrieval-Augmented-Generation-Pipeline, die beliebige Text- und PDF-Dokumente indexiert und präzise Fragen dazu per GPT-3.5-turbo beantwortet. Die Architektur kombiniert semantisches Chunking, OpenAI-Embeddings und ChromaDB als Vektordatenbank.

## Architektur

```
                          Ingestion
  ┌──────────┐   chunk   ┌──────────┐   embed    ┌──────────┐   upsert   ┌──────────┐
  │ Dokument │ ────────► │ Ingestor │ ─────────► │ Embedder │ ─────────► │ ChromaDB │
  │ .txt/.pdf│           │          │            │  ada-002 │            │  (lokal) │
  └──────────┘           └──────────┘            └──────────┘            └──────────┘

                          Query
  ┌──────────┐   embed   ┌──────────┐   search   ┌──────────┐  generate  ┌──────────┐
  │  Frage   │ ────────► │Retriever │ ─────────► │ ChromaDB │ ─────────► │  GPT-3.5 │
  │          │           │          │ ◄───chunks─│          │            │  -turbo  │
  └──────────┘           └──────────┘            └──────────┘            └──────────┘
                              │
                              ▼
                         formatierte
                           Antwort
```

## Voraussetzungen

- Python 3.9+
- OpenAI API Key
- SQLite >= 3.35.0 (für ChromaDB)

## Installation

```bash
pip install -r requirements.txt
```

```bash
export OPENAI_API_KEY="sk-..."
```

## Verwendung

### Dokument indexieren

```bash
python -m src.cli ingest --file data/documents/beispiel.txt --collection handbuch
```

Ausgabe:
```
[1/3] Lade und chunke Dokument: beispiel.txt …
      12 Chunk(s) erstellt.
[2/3] Erstelle Embeddings …
      12 Embedding(s) generiert.
[3/3] Speichere in Collection 'handbuch' …
      Fertig. 12 Chunk(s) gespeichert.
```

### Frage stellen

```bash
python -m src.cli query \
  --question "Wie oft muss das Öl gewechselt werden?" \
  --collection handbuch \
  --top-k 4
```

Ausgabe:
```
============================================================
Antwort:
Der Ölwechsel ist alle 250 Betriebsstunden (monatliche Wartung) durchzuführen. ...

Quellen:
  - data/documents/beispiel.txt

Verwendete Chunks: 3
============================================================
```

## Projektstruktur

```
docuquery-rag/
├── src/
│   ├── ingestor.py      # Dokument laden + tokenbasiertes Chunking (tiktoken)
│   ├── embedder.py      # OpenAI-Embeddings erstellen + ChromaDB-Speicherung
│   ├── retriever.py     # Semantische Suche + GPT-3.5-turbo Antwortgenerierung
│   └── cli.py           # Argparse-CLI (ingest / query)
├── tests/
│   ├── conftest.py      # chromadb-Mock (SQLite-Workaround)
│   ├── test_ingestor.py # 22 Unit-Tests
│   ├── test_embedder.py # 12 Unit-Tests
│   └── test_retriever.py# 6 Unit-Tests
├── data/
│   └── documents/       # Eigene Dokumente hier ablegen
├── .chromadb/           # ChromaDB-Persistenz (auto-generiert, nicht committen)
└── requirements.txt
```

## Tests

```bash
python -m pytest tests/ -v
```

Alle 40 Tests laufen ohne echte API-Aufrufe (vollständig gemockt).

## Konfiguration

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `--collection` | `default` | Name der ChromaDB-Collection |
| `--top-k` | `5` | Maximale Anzahl Chunks für die Antwortgenerierung |
| `SCORE_THRESHOLD` | `0.7` | Mindest-Cosine-Similarity für Chunk-Selektion |
| Chunk-Größe | `512 Tokens` | Konfigurierbar in `DocumentIngestor` |
| Chunk-Overlap | `64 Tokens` | Konfigurierbar in `DocumentIngestor` |

## Lizenz

MIT
