from ollama import chat

def ollama_call(model:str, messages):
    """
    Ollama API call.

    Parameters:
        Model name, full prompt payload
    
    Returns:
        message response from model
    """

    try:
        response = chat(model=model, messages = messages)
        return response.message.content
    except Exception as e:
        print(f"Ollama error: {e}")
        return None
