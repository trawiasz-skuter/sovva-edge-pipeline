import requests
from core.schemas import SovvaPayload

def send_payload(payload: SovvaPayload, api_url: str = "http://192.168.3.222:8000/api/ingest"):
    try:
        response = requests.post(
            api_url, 
            data=payload.model_dump_json(), 
            headers={"Content-Type": "application/json"},
            timeout=5)

        response.raise_for_status()
        print(f"Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Błąd sieci podczas wysyłania: {e}")