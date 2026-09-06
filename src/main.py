from pathlib import Path
from config import settings
from datetime import datetime, timezone
import logging
from video.extractor import get_video_metadata, chunk_video, extract_frames
from ai.prompt_builder import build_prompt, ollama_payload
from ai.base_convert import convert_tobase
from ai.vlm_client import ollama_call
from ai.embedder import response_embedding
from ai.profiles import get_model_profile
from core.builder import build_payload
from network.transmitter import send_payload

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

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting SOVVA Edge Pipeline")
    logger.info(f"Target video file: {settings.video_path}")

    try:
        fps, width, height, frame_count = get_video_metadata(settings.video_path)
        logger.info(
            f"Video metadata loaded: {width}x{height} resolution, {fps:.2f} FPS, {frame_count} total frames"
        )

        all_idxs = chunk_video(
            settings.chunk_t, settings.chunk_frames, settings.overlap, fps, frame_count
        )
        total_chunks = len(all_idxs)
        logger.info(
            f"Video successfully partitioned into {total_chunks} chunk(s) (chunk_duration={settings.chunk_t}s, overlap={settings.overlap}s)"
        )

        stream = extract_frames(settings.video_path, all_idxs)

        profile = get_model_profile("gemma4:e2b")

        previous_caption = None

        for chunk_idx, chunk in enumerate(stream):
            logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks}...")

            chunk_stream = convert_tobase(chunk)
            prompt_text = build_prompt(
                frames_idxs=all_idxs[chunk_idx], 
                fps=fps,
                template=profile.prompt_template,
                previous_caption=previous_caption
            )

            payload = ollama_payload(
                message=prompt_text, 
                chunk_frames=chunk_stream, 
                system_prompt=profile.prompt_template.system_prompt
            )

            logger.debug(f"Chunk {chunk_idx + 1}: Querying VLM model 'gemma4:e2b'")
            model_content = ollama_call(profile=profile, messages=payload)
            previous_caption = model_content
            preview_content = (
                (model_content[:80] + "...")
                if model_content and len(model_content) > 80
                else model_content
            )
            logger.info(
                f'Chunk {chunk_idx + 1}: VLM analysis completed: "{preview_content}"'
            )

            logger.debug(f"Chunk {chunk_idx + 1}: Computing semantic embedding")
            embedding_content = response_embedding(model_content)

            gate_payload = build_payload(
                drone_id="dfg231s",
                timestamp=datetime.now(timezone.utc),
                location=[30.0, 21.3],
                altitude_m=20.3,
                model_name=profile.model_name,
                caption=model_content,
                embedding=embedding_content,
            )

            logger.debug(
                f"Chunk {chunk_idx + 1}: Transmitting payload to ingestion gate"
            )
            send_payload(gate_payload)
            logger.info(
                f"Chunk {chunk_idx + 1}/{total_chunks} successfully processed and transmitted"
            )

        logger.info("SOVVA Edge Pipeline execution finished successfully")

    except Exception as e:
        logger.exception(f"Unhandled error during pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()
