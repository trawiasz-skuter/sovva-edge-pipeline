from tenacity import retry, stop_after_attempt, wait_exponential
from ollama import chat
import logging
from ai.profiles import VLMModelProfile

logger = logging.getLogger(__name__)

class VLMInferenceError(Exception):
    """An exception thrown when the VLM fails to generate a response."""

    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def ollama_call(profile: VLMModelProfile, messages: list[dict]) -> str:
    """
    Ollama API call.

    Parameters:
        Model name, full prompt payload

    Returns:
        message response from model
    """
    options_dict = profile.options.model_dump(exclude_none=True)

    logger.debug(
        f"Initiating Ollama chat request (model='{profile.model_name}' with options: {options_dict}, message_count={len(messages)})"
    )
    try:
        response = chat(
            model=profile.model_name, 
            messages=messages,
            options=options_dict if options_dict else None,
            )
        content = response.message.content
        if not content:
            logger.error(f"Ollama returned an empty response for model='{profile.model_name}'")
            raise VLMInferenceError("Ollama returned empty response")
        logger.debug(
            f"Ollama VLM inference succeeded for model='{profile.model_name}' (response_length={len(content)})"
        )
        return content
    except Exception as e:
        logger.error(f"Ollama VLM call failed for model='{profile.model_name}': {e}")
        raise
