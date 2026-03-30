import asyncio
import time
import json
import httpx

from typing import List, Optional, Dict, Any

from core.config import settings
from core.logging import get_logger

from models.ups import UPSRateRequest, RateQuote
from exceptions.usps_exceptions import USPSAPIError
from clients.usps_client import usps_token_manager

logger = get_logger(__name__)


class USPSAddressService:
  def __init__(self):
    self.client = None

  async def __aenter__(self):
    self.client = httpx.AsyncClient(
      timeout=settings.USPS_REQUEST_TIMEOUT,
      limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self.client:
      await self.client.aclose()

  async def validate_address(self, street_address: str, city: str, state: str) -> Dict[str, str]:
    token = await usps_token_manager.get_token(scope="addresses")

    params = {
      "streetAddress": street_address,
      "city": city,
      "state": state
    }

    headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json"
    }

    logger.info(f"Validating USPS address: {street_address}, {city}, {state}")

    for attempt in range(settings.USPS_MAX_RETRIES):
      try:
        response = await self.client.get(
          settings.USPS_ADDRESS_VALIDATION_URL,
          params=params,
          headers=headers
        )

        if response.status_code == 401:
          error_text = response.text
          logger.error(f"USPS Address API authentication failed (401). Response: {error_text}")
          usps_token_manager.invalidate_token(scope="addresses")
          raise USPSAPIError("Authentication failed", "AUTH_ERROR", 401)

        elif response.status_code != 200:
          error_text = response.text
          logger.error(f"USPS Address API returned status {response.status_code}: {error_text}")

          try:
            error_data = response.json()
            if "error" in error_data:
              error_info = error_data["error"]
              error_msg = error_info.get("message", "Unknown error")
              error_code = error_info.get("code", "")
              raise USPSAPIError(
                f"USPS API error {error_code}: {error_msg}",
                error_code,
                response.status_code
              )
          except json.JSONDecodeError:
            logger.error(f"Failed to parse USPS error response as JSON: {error_text}")

          raise USPSAPIError(f"USPS API error: {response.status_code}", "API_ERROR", response.status_code)

        response_data = response.json()

        address = response_data.get("address", {})
        zip_code = address.get("ZIPCode", "")
        zip_plus4 = address.get("ZIPPlus4")

        result = {"zipCode": zip_code}
        if zip_plus4:
          result["zipPlus4"] = zip_plus4

        logger.info(f"USPS address validation successful: ZIP={zip_code}, ZIP+4={zip_plus4}")
        return result

      except (httpx.TimeoutException, httpx.ConnectError):
        if attempt == settings.USPS_MAX_RETRIES - 1:
          raise USPSAPIError("Connection timeout", "TIMEOUT_ERROR", 503)

        await asyncio.sleep(2 ** attempt)


class USPSRateService:
  def __init__(self):
    self.client = None

  async def __aenter__(self):
    self.client = httpx.AsyncClient(
      timeout=settings.USPS_REQUEST_TIMEOUT,
      limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self.client:
      await self.client.aclose()

  def _is_domestic(self, sender_country: str, destination_country: str) -> bool:
    return sender_country.upper() == "US" and destination_country.upper() == "US"

  def _build_letter_payload(self, request: UPSRateRequest) -> Optional[Dict[str, Any]]:
    if not self._is_domestic(request.sender_country, request.country):
      return None

    weight_oz = max(0.0, float(request.weight)) * 16.0
    if weight_oz <= 0 or weight_oz > 3.5:
      return None

    length = float(request.length or 0)
    height = float(request.height or 0)
    width = float(request.width or 0)

    if length > 0 and height > 0 and width > 0:
      dims = sorted([length, height, width])
      thickness, height, length = dims[0], dims[1], dims[2]
    else:
      length = 6.0
      height = 4.0
      thickness = 0.02

    return {
      "weight": weight_oz,
      "length": length,
      "height": height,
      "thickness": thickness,
      "processingCategory": "LETTERS",
    }

  def _build_rate_payload(self, request: UPSRateRequest) -> Dict[str, Any]:
    is_domestic = self._is_domestic(request.sender_country, request.country)

    if not request.sender_zip:
      raise ValueError("sender_zip is required for USPS rate calculation")

    if not request.zip and is_domestic:
      raise ValueError("zip is required for USPS domestic rate calculation")

    payload = {
      "originZIPCode": request.sender_zip,
      "weight": request.weight,
      "length": max(1, float(request.length)) if request.length > 0 else 1.0,
      "width": max(1, float(request.width)) if request.width > 0 else 1.0,
      "height": max(1, float(request.height)) if request.height > 0 else 1.0,
      "priceType": "COMMERCIAL"
    }

    if is_domestic:
      payload["destinationZIPCode"] = request.zip
    else:
      payload["destinationCountryCode"] = request.country
      if request.zip:
        payload["foreignPostalCode"] = request.zip

    return payload

  async def get_rates(self, request: UPSRateRequest, request_id: str = None) -> List[RateQuote]:
    start_time = time.time()

    if request_id is None:
      request_id = f"usps_rate_{int(time.time() * 1000)}"

    local_request_id = request_id

    try:
      is_domestic = self._is_domestic(request.sender_country, request.country)

      scope = "domestic-prices" if is_domestic else "international-prices"
      token = await usps_token_manager.get_token(scope=scope)

      payload = self._build_rate_payload(request)

      api_url = settings.USPS_DOMESTIC_RATES_URL if is_domestic else settings.USPS_INTERNATIONAL_RATES_URL

      response = await self._make_usps_request(token, payload, api_url, local_request_id)
      quotes = self._parse_rate_response(response, is_domestic)

      if is_domestic:
        letter_payload = self._build_letter_payload(request)
        if letter_payload:
          try:
            letter_response = await self._make_usps_letter_request(
              token,
              letter_payload,
              settings.USPS_LETTER_RATES_URL,
              local_request_id
            )
            quotes.extend(self._parse_letter_rate_response(letter_response))
          except Exception as e:
            logger.warning(f"Letter rate failed: {e}")

      logger.info(f"USPS Rate request {local_request_id} completed")
      return quotes

    except Exception as e:
      logger.error(f"USPS Rate request {local_request_id} failed: {e}")
      raise
