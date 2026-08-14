from ollama import embed

def response_embedding(response_content):
    """
    Model response content embedding

    Parameters:
        Model reponse
    
    Returns:
        Embedding
    """
    response = embed(
        model="all-minilm",
        input=response_content
    )

    return response["embeddings"][0]