import base64
from PIL import Image
import io


def convert_tobase(chunk: list):
    """
    Returns converted image to Base64 format

    Parameters:
        List of images (numpy array)

    Returns:
        List of converted images
    """

    for _, frame in enumerate(chunk):
        img = Image.fromarray(frame)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        yield base64.b64encode(img_bytes).decode("utf-8")
