# 04. Configuration & Environment Management

Source module: [`src/config.py`](../src/config.py)  
Environment file: `.env`

---

## ⚙️ Pydantic BaseSettings Model

Configuration management is strongly-typed and validated using `pydantic-settings`. All settings are defined in a centralized class and automatically mapped to environment variables or values declared in `.env`:

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    video_path: str
    chunk_t: int = Field(validation_alias="TIME_OF_CHUNK")
    chunk_frames: int = Field(validation_alias="EXTRACTED_FRAMES_PER_CHUNK")
    overlap: int = Field(validation_alias="OVERLAP_T")
    api_end_point: str = Field(validation_alias="GATE_END_POINT")
    log_level: str = Field(
        validation_alias="LOG_LEVEL",
        examples=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"],
    )
    log_file_path: str = Field(validation_alias="LOG_FILE_PATH")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
```

---

## 📋 Environment Variables (`.env`)

| Variable | Target Field in `Settings` | Type | Default / Example | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| `VIDEO_PATH` | `video_path` | `str` | `"data/input/videoplayback.mp4"` | Path to the source video file or RTSP stream URL. |
| `TIME_OF_CHUNK` | `chunk_t` | `int` | `10` | Duration of each sliding observation window in seconds. |
| `EXTRACTED_FRAMES_PER_CHUNK` | `chunk_frames` | `int` | `4` | Number of frames sampled uniformly per window for VLM analysis. |
| `OVERLAP_T` | `overlap` | `int` | `2` | Duration of temporal overlap between consecutive chunks in seconds. |
| `GATE_END_POINT` | `api_end_point` | `str` | `"http://192.168.3.222:8000/api/ingest"` | Destination HTTP endpoint of the S.O.V.V.A. Ingestion Gate. |
| `LOG_LEVEL` | `log_level` | `str` | `"INFO"` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LOG_FILE_PATH` | `log_file_path` | `str` | `"logs"` | Directory path where persistent log files are stored. |

---

## 🪵 Dual-Channel Logging Setup

The application configures a robust, multi-target logging system initialized at startup:

```python
log_dir = Path(settings.log_file_path)
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "sovva_edge.log"

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
```

- **FileHandler:** Ensures persistent on-disk recording for post-mission telemetry diagnostics.
- **StreamHandler:** Provides real-time operator console feedback with chunk processing progress.

---

## 🎯 Edge Parameter Tuning Guidelines

1. **Chunk Duration (`TIME_OF_CHUNK`):**
   - *Too short (< 3s):* The VLM lacks temporal context to infer activity progression or vehicle heading.
   - *Too long (> 25s):* Frame sampling becomes too sparse, increasing the risk of missing brief actions.
   - *Recommendation:* **$6 - 12\text{ seconds}$**.

2. **Extracted Frames per Chunk (`EXTRACTED_FRAMES_PER_CHUNK`):**
   - *Higher values (6-12):* Finer motion granularity, but increases VLM prompt token count and inference latency.
   - *Lower values (3-4):* Lowest compute overhead, ideal for edge devices with limited VRAM.
   - *Recommendation:* **$4 - 6\text{ frames}$** for edge 2B-7B parameter models.

3. **Temporal Overlap (`OVERLAP_T`):**
   - Prevents activity boundary truncation when an event occurs across chunk splits.
   - *Recommendation:* Set to **$15\% - 25\%$** of `TIME_OF_CHUNK` (e.g., $2\text{s}$ overlap for a $10\text{s}$ chunk).
