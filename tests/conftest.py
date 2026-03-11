"""
Pytest-Konfiguration für das Projekt.

ChromaDB setzt SQLite >= 3.35.0 voraus; dieses System hat 3.34.0.
Da alle Embedder-Tests die externe API mocken, ersetzen wir das
chromadb-Modul bereits beim Import durch einen MagicMock, bevor
src.embedder geladen wird. Dadurch schlägt der sqlite-Check nie an.
"""

import sys
from unittest.mock import MagicMock

# Muss vor dem ersten Import von src.embedder stehen.
sys.modules.setdefault("chromadb", MagicMock())
