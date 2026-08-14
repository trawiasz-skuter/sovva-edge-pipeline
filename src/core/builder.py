from datetime import datetime
from typing import List, Dict, Any
from .schemas import SovvaPayload, TelemetryData

def build_payload(
    drone_id: str,
    timestamp: datetime, 
    location: List[float],
    altitude_m: float,
    model_name: str,
    caption: str,
    embedding: List[float]
) -> SovvaPayload:
    """
    Mapuje surowe słowniki i zmienne na zwalidowany obiekt Pydantic.
    """
    telemetry_obj = TelemetryData(
        drone_id=drone_id,
        timestamp=timestamp,
        location=location,
        altitude=altitude_m
    )

    return SovvaPayload(
        model_name=model_name,
        caption=caption,
        embedding=embedding
    )