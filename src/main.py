from video.extractor import get_video_metadata, chunk_video, extract_frames
from ai.prompt_builder import build_prompt, ollama_payload
from ai.base_convert import convert_tobase
from ai.vlm_client import ollama_call
from ai.embedder import response_embedding
from core.builder import build_payload
from network.transmitter import send_payload
from config import settings
from datetime import datetime, timezone


def main():
    fps, _, _, frame_count = get_video_metadata(settings.video_path)
    all_idxs = chunk_video(
        settings.chunk_t, settings.chunk_frames, settings.overlap, fps, frame_count
    )
    stream = extract_frames(settings.video_path, all_idxs)

    for chunk_idx, chunk in enumerate(stream):
        chunk_stream = convert_tobase(chunk)
        prompt_text = build_prompt(frames_idxs=all_idxs[chunk_idx], fps=fps)
        payload = ollama_payload(prompt_text, chunk_stream)
        model_content = ollama_call("gemma4:e2b", payload)
        embedding_content = response_embedding(model_content)
        gate_payload = build_payload(
            drone_id="dfg231s",
            timestamp=datetime.now(timezone.utc),
            location=[30.0, 21.3],
            altitude_m=20.3,
            model_name="gemma4:e2b",
            caption=model_content,
            embedding=embedding_content,
        )
        send_payload(gate_payload)


if __name__ == "__main__":
    main()
