import requests
import time
from datetime import datetime, timezone
from typing import Dict, Any

from auth.fedex_oauth import FedexOauth
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

class FedexService:
  def __init__(self):
    self.auth = FedexOauth()

  def get_rates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = f"fedex_rate_{int(time.time() * 1000)}"
    start_time = time.time()

    logger.info(f"FedEx Rate request {request_id} started, payload keys: {list(payload.keys())}")

    try:
      url = f"{settings.FEDEX_BASE_URL}/rate/v1/rates/quotes"

      weight = payload['requestedShipment']["requestedPackageLineItems"][0]["weight"]
      validated_weight = max(weight["value"], 1)
      if validated_weight != weight["value"]:
        logger.info(f"Adjusted weight from {weight['value']} to {validated_weight} (FedEx minimum 1 lb)")
      weight["value"] = validated_weight

      payload = {
        **payload,
        "accountNumber": {
          "value": settings.FEDEX_ACCOUNT_NUMBER
        }
      }

      result = self._make_request_with_retry(url, payload, request_id)

      processing_time = (time.time() - start_time) * 1000
      logger.info(f"FedEx Rate request {request_id} completed in {processing_time:.2f}ms")
      return result

    except Exception as e:
      processing_time = (time.time() - start_time) * 1000
      logger.error(f"FedEx Rate request {request_id} failed after {processing_time:.2f}ms: {str(e)}")
      raise

  def _make_request_with_retry(self, url: str, payload: Dict[str, Any], request_id: str, retries=3) -> Dict[str, Any]:
    attempt = 0

    while attempt < retries:
      token = self.auth.get_token()
      headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
      }

      try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code < 400:
          return response.json()

        logger.warning(f"FedEx API returned status {response.status_code} on attempt {attempt+1} for request {request_id}")

        if response.status_code == 401:
          logger.info("FedEx token expired, invalidating token")
          self.auth._access_token = None
          attempt += 1
          continue

        if response.status_code >= 500:
          logger.info(f"Server error, retrying after backoff for request {request_id}")
          attempt += 1
          time.sleep(2 ** attempt)
          continue

        logger.error(f"FedEx API error: {response.status_code} - {response.text}")
        raise Exception({
          "status_code": response.status_code,
          "response": response.json()
        })

      except requests.exceptions.RequestException as e:
        attempt += 1
        logger.warning(f"Request exception on attempt {attempt} for {request_id}: {str(e)}")
        time.sleep(2 ** attempt)

    logger.error(f"FedEx API failed after {retries} attempts for request {request_id}")
    raise Exception("FedEx API failed after retries")
