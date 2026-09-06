from tenacity import retry, stop_after_attempt, wait_exponential
from ollama import chat
import logging

logger = logging.getLogger(__name__)

class VLMInferenceError(Exception):
    """An exception thrown when the VLM fails to generate a response."""
    pass

def ollama_call(model: str, messages: list[dict]) -> str:
    """
    Ollama API call.

    Parameters:
        Model name, full prompt payload

    Returns:
        message response from model
    """
    logger.debug(f"Initiating Ollama chat request (model='{model}', message_count={len(messages)})")
    try:
        response = chat(model=model, messages=messages)
        content = response.message.content
        if not content:
            logger.error(f"Ollama returned an empty response for model='{model}'")
            raise VLMInferenceError("Ollama returned empty response")
        logger.debug(f"Ollama VLM inference succeeded for model='{model}' (response_length={len(content)})")
        return content
    except Exception as e:
        logger.error(f"Ollama VLM call failed for model='{model}': {e}")
        raise

