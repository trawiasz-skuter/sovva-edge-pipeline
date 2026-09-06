import logging
import requests
from config import settings
from core.schemas import UavFlightDataPayload

logger = logging.getLogger(__name__)

session = requests.Session()

def send_payload(payload: UavFlightDataPayload, api_url: str = settings.api_end_point):
    logger.debug(f"Preparing to send payload to {api_url}")
    try:
        response = session.post(
            api_url,
            data=payload.model_dump_json(),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        response.raise_for_status()
        logger.info(
            f"Payload successfully delivered to {api_url} (HTTP {response.status_code})"
        )

    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error delivering payload to {api_url}: {e.response.status_code} - {e.response.text}"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error communicating with {api_url}: {e}")
