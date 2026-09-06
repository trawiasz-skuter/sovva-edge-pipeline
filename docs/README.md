# S.O.V.V.A. Edge Pipeline Architecture Documentation

This directory contains the complete technical and architectural documentation for the **S.O.V.V.A. Edge Pipeline** (Semantic Observation / Video Vision Analytics on Edge).

The documentation provides thorough context for system engineers, software developers, and autonomous agent systems analyzing or extending this codebase.

---

## 🗺️ Documentation Map

| File | Scope | Summary |
| :--- | :--- | :--- |
| [01_system_overview.md](01_system_overview.md) | **System Architecture** | Project goals, high-level architecture, end-to-end Mermaid dataflow, technology stack, and module tree. |
| [02_video_processing.md](02_video_processing.md) | **Video Processing** | Mathematical formulation of sliding window chunking with temporal overlap, uniform sampling, and generator streaming. |
| [03_vlm_inference.md](03_vlm_inference.md) | **AI & VLM Inference** | Model profile registry, active prompting with timestamps, rolling scene memory, retry policies, and CPU FastEmbed integration. |
| [04_configuration.md](04_configuration.md) | **Configuration & Environment** | Pydantic Settings, environment variables (`.env`), logging parameters, and parameter tuning guidelines. |
| [05_data_flow_and_contracts.md](05_data_flow_and_contracts.md) | **Data Flow & Contracts** | Stage-by-stage type transformations, Pydantic domain models, and JSON schema contracts for the Ingestion Gate & OpenSearch. |
| [06_ecosystem_and_roadmap.md](06_ecosystem_and_roadmap.md) | **Ecosystem & Roadmap** | UAV surveillance ecosystem context, telemetry synchronization, completed milestones, and future development phases. |
| [07_code_review_and_production_improvements.md](07_code_review_and_production_improvements.md) | **Code Review & Production Audit** | In-depth engineering audit, benchmark results, resilience analysis, and best practice recommendations. |

---

## ⚡ Quick Execution Guide

```bash
# 1. Environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start Ollama and download vision model
ollama pull gemma4:e2b

# 3. Configure environment
cp .env.example .env  # Or populate .env with required keys

# 4. Execute edge pipeline
python src/main.py
```
