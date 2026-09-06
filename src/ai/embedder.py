import logging
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder_instance: TextEmbedding | None = None


def get_embedder(model_name: str = DEFAULT_MODEL_NAME) -> TextEmbedding:
    """
    Returns the embedding model instance (Lazy Singleton).
    Ensures the ONNX model is loaded into memory only once during pipeline startup.
    """
    global _embedder_instance
    if _embedder_instance is None:
        logger.info(f"Initializing local CPU embedder ({model_name})...")
        _embedder_instance = TextEmbedding(model_name=model_name)
        logger.info("Local CPU embedder loaded successfully.")
    return _embedder_instance


def response_embedding(
    response_content: str, model_name: str = DEFAULT_MODEL_NAME
) -> list[float]:
    """
    Generates an embedding vector (384-D) locally on the CPU using fastembed.

    Parameters:
        response_content: Response text from the VLM model
        model_name: Identifier of the ONNX-optimized model

    Returns:
        384-element list of floats representing the semantic embedding vector
    """
    if not response_content:
        logger.warning(
            "Empty response received for embedding. Generating vector for empty text."
        )
        response_content = ""

    logger.debug(
        f"Generating embedding locally on CPU (model='{model_name}', text_length={len(response_content)})"
    )
    try:
        embedder = get_embedder(model_name)
        # fastembed yields an iterator of numpy ndarrays
        embeddings = list(embedder.embed([response_content]))
        embedding_vector: list[float] = embeddings[0].tolist()

        logger.debug(
            f"Embedding generated successfully on CPU (dimension={len(embedding_vector)})"
        )
        return embedding_vector

    except Exception as e:
        logger.error(f"Failed to generate embedding on CPU: {e}")
        raise
