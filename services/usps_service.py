import asyncio
import time
import json
import httpx

from typing import List, Optional, Dict, Any

from core.config import settings
from core.logging import get_logger

from schemas.ups import UPSRateRequest, RateQuote
from core.exceptions.usps import USPSAPIError
from auth.usps_oauth import USPSOauth

logger = get_logger(__name__)

usps_oauth = USPSOauth()


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
    token = usps_oauth.get_token(scope="addresses")

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
          usps_oauth.invalidate_token(scope="addresses")
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

    logger.info(f"[LETTER-RATES] USPS LetterRatesQuery request: {request}")
    weight_oz = max(0.0, float(request.weight)) * 16.0
    logger.info(f"[LETTER-RATES] USPS LetterRatesQuery weight: {weight_oz} oz")
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

    payload = {
      "weight": weight_oz,
      "length": length,
      "height": height,
      "thickness": thickness,
      "processingCategory": "LETTERS",
    }
    logger.info(f"[LETTER-RATES] USPS LetterRatesQuery payload: {payload}")
    return payload

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
      token = usps_oauth.get_token(scope=scope)

      payload = self._build_rate_payload(request)
      api_url = settings.USPS_DOMESTIC_RATES_URL if is_domestic else settings.USPS_INTERNATIONAL_RATES_URL
      logger.info(f"Making USPS API request ({'domestic' if is_domestic else 'international'}) to {api_url}")

      response = await self._make_usps_request(token, payload, api_url, local_request_id)
      quotes = self._parse_rate_response(response, is_domestic)

      if is_domestic:
        letter_payload = self._build_letter_payload(request)
        if letter_payload:
          logger.info(f"[LETTER-RATES] [{local_request_id}] Eligible for First-Class Mail - calling letter-rates API")
          try:
            letter_response = await self._make_usps_letter_request(token, letter_payload, settings.USPS_LETTER_RATES_URL, local_request_id)
            letter_quotes = self._parse_letter_rate_response(letter_response)
            logger.info(f"[LETTER-RATES] [{local_request_id}] Added {len(letter_quotes)} First-Class Mail quote(s) to results")
            quotes.extend(letter_quotes)
          except USPSAPIError as e:
            logger.warning(f"[LETTER-RATES] [{local_request_id}] First-Class Mail request failed: {e}")
          except Exception as e:
            logger.warning(f"[LETTER-RATES] [{local_request_id}] Unexpected error fetching letter rates: {e}")
        else:
          logger.info(f"[LETTER-RATES] [{local_request_id}] Skipped - weight={float(request.weight)*16:.2f}oz (must be >0 and <=3.5oz)")

      processing_time = (time.time() - start_time) * 1000
      logger.info(f"USPS Rate request {local_request_id} completed in {processing_time:.2f}ms, {len(quotes)} quotes returned")
      return quotes

    except Exception as e:
      processing_time = (time.time() - start_time) * 1000
      logger.error(f"USPS Rate request {local_request_id} failed after {processing_time:.2f}ms: {str(e)}")
      raise

  async def _make_usps_request(self, token: str, payload: Dict[str, Any], api_url: str, request_id: str) -> Dict[str, Any]:
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {token}",
      "User-Agent": "USPS-Rating-Microservice/1.0"
    }

    logger.info(f"USPS API request [{request_id}]: {payload.get('originZIPCode')} -> {payload.get('destinationZIPCode', payload.get('destinationCountryCode'))}")

    for attempt in range(settings.USPS_MAX_RETRIES):
      try:
        response = await self.client.post(api_url, json=payload, headers=headers)
        logger.info(f"USPS API response [{request_id}]: status={response.status_code}, body={response.text}")

        if response.status_code == 401:
          logger.error(f"USPS API authentication failed (401) [{request_id}]")
          usps_oauth.invalidate_token(scope="domestic-prices" if "domestic" in api_url else "international-prices")
          raise USPSAPIError("Authentication failed", "AUTH_ERROR", 401)
        elif response.status_code != 200:
          logger.error(f"USPS API returned status {response.status_code} [{request_id}]: {response.text}")
          raise USPSAPIError("USPS API error", "API_ERROR", response.status_code)

        return response.json()

      except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning(f"USPS API connection error [{request_id}] attempt {attempt+1}: {e}")
        if attempt == settings.USPS_MAX_RETRIES - 1:
          raise USPSAPIError("Connection timeout", "TIMEOUT_ERROR", 503)
        await asyncio.sleep(2 ** attempt)

  async def _make_usps_letter_request(self, token: str, payload: Dict[str, Any], api_url: str, request_id: str) -> Dict[str, Any]:
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {token}",
      "User-Agent": "USPS-Rating-Microservice/1.0"
    }

    logger.info(f"[LETTER-RATES] [{request_id}] Sending request to {api_url}, payload: {json.dumps(payload)}")

    for attempt in range(settings.USPS_MAX_RETRIES):
      try:
        response = await self.client.post(api_url, json=payload, headers=headers)
        logger.info(f"[LETTER-RATES] [{request_id}] Response status={response.status_code}, body={response.text}")

        if response.status_code == 401:
          logger.error(f"[LETTER-RATES] [{request_id}] Authentication failed (401)")
          usps_oauth.invalidate_token(scope="domestic-prices")
          raise USPSAPIError("Authentication failed", "AUTH_ERROR", 401)
        elif response.status_code != 200:
          raise USPSAPIError("USPS Letter API error", "API_ERROR", response.status_code)

        response_data = response.json()
        logger.info(f"[LETTER-RATES] [{request_id}] Success - parsed response: {json.dumps(response_data)}")
        return response_data

      except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning(f"[LETTER-RATES] [{request_id}] Connection error on attempt {attempt+1}: {e}")
        if attempt == settings.USPS_MAX_RETRIES - 1:
          raise USPSAPIError("Connection timeout", "TIMEOUT_ERROR", 503)
        await asyncio.sleep(2 ** attempt)

  def _parse_rate_response(self, response_data: Dict[str, Any], is_domestic: bool) -> List[RateQuote]:
    quotes = []
    logger.info(f"Parsing USPS response for {'domestic' if is_domestic else 'international'}")
    rate_options = response_data.get("rateOptions", [])
    if not rate_options:
      logger.warning(f"No rateOptions found in USPS response: {response_data}")
      return quotes
    if not isinstance(rate_options, list):
      rate_options = [rate_options]
    logger.info(f"Found {len(rate_options)} rate options to process")

    for rate_option in rate_options:
      rates = rate_option.get("rates", [])
      if not isinstance(rates, list):
        rates = [rates]

      total_base_price = rate_option.get("totalBasePrice", 0)

      for rate in rates:
        mail_class = rate.get("mailClass", "")
        rate_indicator = rate.get("rateIndicator", "")
        product_name = rate.get("productName", mail_class)
        service_code = f"{mail_class}|{rate_indicator}" if rate_indicator else mail_class
        context = "Domestic" if is_domestic else "International"
        quote = RateQuote(
          service_code=service_code,
          service_name=product_name,
          total_charge=float(total_base_price),
          currency_code="USD",
          context=context
        )
        quotes.append(quote)
        logger.info(f"Added USPS quote: {product_name} ({service_code}) - ${total_base_price}")

    logger.info(f"Successfully parsed {len(quotes)} quotes from USPS response")
    return quotes

  def _parse_letter_rate_response(self, response_data: Dict[str, Any]) -> List[RateQuote]:
    quotes: List[RateQuote] = []
    logger.info(f"[LETTER-RATES] Parsing letter rate response")
    total_base_price = response_data.get("totalBasePrice", 0)
    rates = response_data.get("rates", [])
    if not isinstance(rates, list):
      rates = [rates]
    logger.info(f"[LETTER-RATES] totalBasePrice={total_base_price}, rates count={len(rates)}")

    for rate in rates:
      product_name = rate.get("description", "First-Class Mail")
      mail_class = rate.get("mailClass", "FIRST-CLASS_MAIL")
      price = float(total_base_price or rate.get("price", 0))
      quote = RateQuote(
        service_code=mail_class,
        service_name=product_name,
        total_charge=price,
        currency_code="USD",
        context="Domestic"
      )
      quotes.append(quote)
      logger.info(f"[LETTER-RATES] Parsed letter rate: mailClass={mail_class}, description={product_name}, price=${price}")

    logger.info(f"[LETTER-RATES] Finished parsing - {len(quotes)} First-Class Mail quote(s) produced")
    return quotes
