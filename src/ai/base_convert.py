import logging
import cv2 as cv
import base64
from typing import Iterator
import numpy as np

logger = logging.getLogger(__name__)

def convert_tobase(chunk: list[np.ndarray], quality: int = 85) -> Iterator[str]:
    """
    Returns converted image to Base64 format

    Parameters:
        List of images (numpy array)

    Returns:
        List of converted images
    """
    logger.debug(f"Converting {len(chunk)} frame(s) to Base64 JPEG (quality={quality})")
    encode_params = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    for idx, frame in enumerate(chunk):
        bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        success, buffer = cv.imencode(".jpg", bgr_frame, encode_params)
        if success:
            yield base64.b64encode(buffer.tobytes()).decode("utf-8")
        else:
            logger.warning(f"Failed to encode frame at index {idx} to JPEG")