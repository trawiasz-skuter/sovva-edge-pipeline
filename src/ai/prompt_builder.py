import datetime
import logging
from ai.schemas import ModelPromptTemplate

logger = logging.getLogger(__name__)


def build_prompt(
    frames_idxs: list,
    fps: float,
    template: ModelPromptTemplate,
    previous_caption: str | None = None,
) -> tuple[str, list[float]]:
    """
    This method provides active prompting allowing model
    for better context understanding of the video providing
    timestamps of each frame.

    Parametrs:
        Frame number in the whole video, frames per
        second of the video

    Returns:
        Text of the prompt
    """
    logger.debug(
        f"Building temporal prompt for {len(frames_idxs)} frame(s) at {fps:.2f} FPS"
    )

    timestamps_list = []
    lines = [template.prefix.strip(), ""]

    if previous_caption:
        lines.append(f'Context from immediately preceding scene: "{previous_caption}"')
        lines.append("")

    for i, j in enumerate(frames_idxs, 1):
        raw_timestamp = round(j / fps, 3)
        timestamp = str(datetime.timedelta(seconds=raw_timestamp))
        lines.append(f"Image {i}: timestamp {timestamp}")
        timestamps_list.append(raw_timestamp)

    lines.append("")
    lines.append(template.suffix.strip())
    final_message = "\n".join(lines)

    return final_message, timestamps_list


def ollama_payload(message: str, chunk_frames, system_prompt: str | None = None):
    """
    Model input payload creator.
    TO DO: Model options, system prompt

    Parameters:
        text prompt, prepared frame data

    Returns:
        message payload
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {"role": "user", "content": message, "images": [img for img in chunk_frames]}
    )

    return messages
