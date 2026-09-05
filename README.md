# S.O.V.V.A. Edge Pipeline

**S.O.V.V.A.** (Semantic Observation / Video Vision Analytics) to brzegowy pipeline przetwarzania materiałów wideo z wykorzystaniem modeli wizyjno-językowych (VLM).

---

## 📚 Dokumentacja Architektury

Kompletna dokumentacja techniczna, diagramy przepływu oraz specyfikacja modułów znajdują się w katalogu [`docs/`](file:///Users/jakubmatkowski/Dokumenty/PW_repos/sovva-edge-pipeline/docs/README.md):

- [docs/01_system_overview.md](file:///Users/jakubmatkowski/Dokumenty/PW_repos/sovva-edge-pipeline/docs/01_system_overview.md) – Architektura systemu, diagram Mermaid, struktura plików i stos technologiczny.
- [docs/02_video_processing.md](file:///Users/jakubmatkowski/Dokumenty/PW_repos/sovva-edge-pipeline/docs/02_video_processing.md) – Chunking wideo (sliding window + overlap), próbkowanie klatek i generatory.
- [docs/03_vlm_inference.md](file:///Users/jakubmatkowski/Dokumenty/PW_repos/sovva-edge-pipeline/docs/03_vlm_inference.md) – Konwersja Base64, budowanie promptu ze znacznikami czasu i integracja z Ollama/MLX.
- [docs/04_configuration.md](file:///Users/jakubmatkowski/Dokumenty/PW_repos/sovva-edge-pipeline/docs/04_configuration.md) – Konfiguracja Pydantic Settings i zmienne środowiskowe `.env`.
- [docs/05_data_flow_and_contracts.md](file:///Users/jakubmatkowski/Dokumenty/PW_repos/sovva-edge-pipeline/docs/05_data_flow_and_contracts.md) – Schematy danych i kontrakty integracyjne z bramką danych/OpenSearch.
- [docs/06_ecosystem_and_roadmap.md](file:///Users/jakubmatkowski/Dokumenty/PW_repos/sovva-edge-pipeline/docs/06_ecosystem_and_roadmap.md) – Kontekst ekosystemu UAV, telemetria i dalszy rozwój projektu.
- [docs/07_code_review_and_production_improvements.md](file:///Users/jakubmatkowski/Dokumenty/PW_repos/sovva-edge-pipeline/docs/07_code_review_and_production_improvements.md) – Obiektywny Code Review, audyt modułów, dobre praktyki produkcyjne i ewaluacja inżynierska.

---

## 🚀 Uruchomienie

```bash
# Instalacja zależności
pip install -r requirements.txt

# Pobranie modelu VLM
ollama pull gemma4:e2b

# Uruchomienie pipeline'u
python src/main.py
```