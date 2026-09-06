import logging
from datetime import datetime
from typing import List, Dict, Any
from .schemas import SovvaPayload, TelemetryData, UavFlightDataPayload

logger = logging.getLogger(__name__)


def build_payload(
    drone_id: str,
    timestamp: datetime,
    location: list[float],
    altitude_m: float,
    model_name: str,
    caption: str,
    embedding: list[float],
) -> UavFlightDataPayload:
    """
    Maps raw dictionaries and variables to validate Pydantic object.
    """
    logger.debug(f"Constructing flight data payload (drone_id='{drone_id}', model='{model_name}')")
    telemetry_obj = TelemetryData(
        drone_id=drone_id, timestamp=timestamp, location=location, altitude=altitude_m
    )

    vlm_obj = SovvaPayload(model_name=model_name, caption=caption, embedding=embedding)

    return UavFlightDataPayload(model_data=vlm_obj, metrics=telemetry_obj)
