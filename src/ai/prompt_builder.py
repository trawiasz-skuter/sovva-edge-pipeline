import datetime

def build_prompt(frames_idxs:list, fps:float) -> str:
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

    lines = [
        "You are given a sequence of images.",
        "The images are provided in the exact order listed below.",
        "Image N corresponds to the N-th image in the input.",
        "",
    ]

    for i, j in enumerate(frames_idxs, 1):
        raw_timestamp = j / fps
        timestamp = str(datetime.timedelta(seconds=raw_timestamp))
        lines.append(f"Image {i}: timestamp {timestamp}")

    lines.extend([
        "",
        "Analyze the sequence in chronological order.",
        "For each image:",
        "- describe the scene,",
        "- identify changes from the previous image,",
        "- infer the ongoing activity.",
    ])

    return "\n".join(lines)

def ollama_payload(message: str, chunk_frames):
    """
    Model input payload creator. 
    TO DO: Model options, system prompt
    
    Parameters: 
        text prompt, prepared frame data
    
    Returns: 
        message payload
    """

    messages = [{
        "role": "user",
        "content": message,
        "images": [img for img in chunk_frames]
    }]

    return messages