from ollama import embed
import logging

logger = logging.getLogger(__name__)

def response_embedding(response_content: str) -> list[float]:
    """
    Model response content embedding

    Parameters:
        Model response text

    Returns:
        Embedding vector
    """
    logger.debug(f"Requesting text embedding via Ollama (model='all-minilm', text_length={len(response_content) if response_content else 0})")
    try:    
        response = embed(model="all-minilm", input=response_content)
        embedding = response["embeddings"][0]
        logger.debug(f"Generated text embedding successfully (dimension={len(embedding)})")
        return embedding
    
    except Exception as e:
        logger.error(f"Ollama embedding call failed: {e}")
        raise
