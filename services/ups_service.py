import time
import re
import json
import asyncio
import httpx

from typing import List, Optional, Dict, Any

from core.config import settings
from core.logging import get_logger
from core.constants.ups import UPSConstants
from core.exceptions import UPSAPIError

from schemas.ups import UPSRateRequest, RateQuote
from auth.ups_oauth import UPSOauth


logger = get_logger(__name__)


class UPSRatingService:
  def __init__(self):
    self.client = None
    self.auth = UPSOauth()
  
  async def __aenter__(self):
    self.client = httpx.AsyncClient(
      timeout=settings.UPS_REQUEST_TIMEOUT,
      limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    return self
  
  async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self.client:
      await self.client.aclose()
  
  def _get_service_id(self, service_code: str, context: str) -> Optional[int]:
    """Get service ID for the given service code and context.
    
    This method returns None to match the frontend's UPSMicroserviceResponse behavior,
    where service_id is explicitly set to None and looked up later by the frontend.
    """
    # Return None for frontend compatibility
    return None
  
  async def get_rates(self, request: UPSRateRequest, request_id: str = None, shop_rates: bool = True) -> List[RateQuote]:
    """Get shipping rates from UPS API"""
    start_time = time.time()
    
    if request_id is None:
      request_id = f"ups_rate_{int(time.time() * 1000)}"
    
    local_request_id = request_id
    
    try:
      request.validate_required_fields()
      
      token = self.auth.get_token()
      
      payload = self._build_rate_payload(request, shop_rates, local_request_id)
      
      response = await self._make_ups_request(token, payload, local_request_id)
      
      context = UPSConstants.get_context(request.sender_country, request.country)
      quotes = self._parse_rate_response(
        response,
        request.sender_country,
        request.country,
        context,
        request.service
      )
      
      processing_time = (time.time() - start_time) * 1000
      logger.info(f"Rate request {local_request_id} completed in {processing_time:.2f}ms, {len(quotes)} quotes returned")
      
      return quotes
      
    except Exception as e:
      processing_time = (time.time() - start_time) * 1000
      logger.error(f"Rate request {local_request_id} failed after {processing_time:.2f}ms: {str(e)}")
      raise
  
  def _build_rate_payload(self, request: UPSRateRequest, shop_rates: bool = True, request_id: str = None) -> Dict[str, Any]:
    """Build UPS API request payload"""
    
    service_code = UPSConstants.SERVICES.get(request.service, "03")
    
    request_type = "Shop"
    
    if request.is_domestic_us():
      if not request.sender_zip or not request.zip:
        raise UPSAPIError("Postal codes are required for US domestic shipments", "MISSING_POSTAL_CODE", 400)
    elif request.is_international():
      if not request.sender_zip and not request.zip:
        logger.warning("At least one postal code recommended for international shipments")
    
    shipper_postal_code = request.sender_zip or ""
    shipto_postal_code = request.zip or ""
    
    if request.is_domestic_us():
      if not shipper_postal_code or not shipto_postal_code:
        raise UPSAPIError("Valid postal codes required for US domestic shipments", "INVALID_POSTAL_CODE", 400)
      
      zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
      
      if not zip_pattern.match(shipper_postal_code.strip()):
        digits_only = re.sub(r'\D', '', shipper_postal_code)
        if len(digits_only) >= 5:
          shipper_postal_code = digits_only[:5]
        else:
          raise UPSAPIError(f"Invalid sender ZIP code format: '{request.sender_zip}'", "INVALID_POSTAL_CODE", 400)
      
      if not zip_pattern.match(shipto_postal_code.strip()):
        digits_only = re.sub(r'\D', '', shipto_postal_code)
        if len(digits_only) >= 5:
          shipto_postal_code = digits_only[:5]
        else:
          raise UPSAPIError(f"Invalid destination ZIP code format: '{request.zip}'", "INVALID_POSTAL_CODE", 400)
    
    elif request.is_international():
      if shipper_postal_code:
        if request.sender_country == "US":
          zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
          if not zip_pattern.match(shipper_postal_code.strip()):
            digits_only = re.sub(r'\D', '', shipper_postal_code)
            if len(digits_only) >= 5:
              shipper_postal_code = digits_only[:5]
        else:
          shipper_postal_code = shipper_postal_code.strip()
      
      if shipto_postal_code:
        if request.country == "US":
          zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
          if not zip_pattern.match(shipto_postal_code.strip()):
            digits_only = re.sub(r'\D', '', shipto_postal_code)
            if len(digits_only) >= 5:
              shipto_postal_code = digits_only[:5]
        else:
          shipto_postal_code = shipto_postal_code.strip()
    
    shipper_address = {
      "PostalCode": shipper_postal_code,
      "CountryCode": request.sender_country
    }
    
    if request.sender_city:
      shipper_address["City"] = request.sender_city
    if request.sender_state:
      shipper_address["StateProvinceCode"] = request.sender_state
    
    if request.is_domestic_us():
      if not request.sender_state or not request.state:
        logger.warning("Missing state information for US domestic shipment - may cause UPS error 111100")
    
    shipto_address = {
      "PostalCode": shipto_postal_code,
      "CountryCode": request.country
    }
    
    if request.city:
      shipto_address["City"] = request.city
    if request.state:
      shipto_address["StateProvinceCode"] = request.state
    
    validated_weight = max(1.0, request.weight)
    if validated_weight != request.weight:
      logger.info(f"Weight adjusted from {request.weight} to {validated_weight} (UPS minimum 1.0 lb)")
    
    package_weight = {
      "Weight": str(validated_weight),
      "UnitOfMeasurement": {
        "Code": "LBS"
      }
    }
    
    if request.package not in UPSConstants.PACKAGES:
      logger.warning(f"Invalid package type '{request.package}', using 'your_packaging'")
      package_type_code = UPSConstants.PACKAGES["your_packaging"]
    else:
      package_type_code = UPSConstants.PACKAGES[request.package]
    
    package_data = {
      "PackagingType": {
        "Code": package_type_code
      },
      "PackageWeight": package_weight
    }
    
    logger.info(f"Package: {request.package} (code: {package_type_code}), Weight: {validated_weight} LBS")
    
    if request.length > 0 or request.width > 0 or request.height > 0:
      validated_length = max(1, int(request.length)) if request.length > 0 else 1
      validated_width = max(1, int(request.width)) if request.width > 0 else 1
      validated_height = max(1, int(request.height)) if request.height > 0 else 1
      
      package_data["Dimensions"] = {
        "UnitOfMeasurement": {
          "Code": "IN"
        },
        "Length": str(validated_length),
        "Width": str(validated_width),
        "Height": str(validated_height)
      }
    
    shipment = {
      "Shipper": {
        "Address": shipper_address
      },
      "ShipTo": {
        "Address": shipto_address
      },
      "Package": [package_data]
    }
    
    if request.service != "all":
      service_code = UPSConstants.SERVICES.get(request.service)
      if service_code:
        shipment["Service"] = {
          "Code": service_code
        }
    
    if request.insured_value > 0:
      shipment["Package"][0]["PackageServiceOptions"] = {
        "InsuredValue": {
          "CurrencyCode": request.currency_code,
          "MonetaryValue": str(request.insured_value)
        }
      }
    
    pickup_code = UPSConstants.PICKUP_TYPES.get(request.pickup_type, "01")
    customer_code = UPSConstants.CUSTOMER_TYPES.get(request.customer_type, "01")
    
    if request.service != "all":
      shipment["ShipmentRatingOptions"] = {
        "NegotiatedRatesIndicator": ""
      }
    
    if request.service == "all":
      request_option = "Shop"
    else:
      request_option = "Rate"
    
    payload = {
      "RateRequest": {
        "Request": {
          "RequestOption": request_option,
          "TransactionReference": {
            "CustomerContext": request_id or "UPS_Rate_Request"
          }
        },
        "PickupType": {
          "Code": pickup_code
        },
        "CustomerClassification": {
          "Code": customer_code
        },
        "Shipment": shipment
      }
    }
    
    return payload
  
  async def _make_ups_request(self, token: str, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {token}",
      "User-Agent": "UPS-Rating-Microservice/1.0"
    }
    
    request_option = payload['RateRequest']['Request']['RequestOption']
    if request_option == "Shop":
      api_url = settings.UPS_RATE_URL.replace('/Rate', '/Shop')
    else:
      api_url = settings.UPS_RATE_URL
    
    logger.info(f"Making UPS API request to: {api_url}")
    logger.info(f"Request summary: {payload['RateRequest']['Shipment']['Shipper']['Address']['CountryCode']} -> {payload['RateRequest']['Shipment']['ShipTo']['Address']['CountryCode']}")
    
    shipper_postal = payload['RateRequest']['Shipment']['Shipper']['Address']['PostalCode']
    shipto_postal = payload['RateRequest']['Shipment']['ShipTo']['Address']['PostalCode']
    
    if not shipper_postal or not shipto_postal:
      logger.warning("Empty postal codes detected - this may cause UPS error 111100")
    
    for attempt in range(settings.UPS_MAX_RETRIES):
      try:
        response = await self.client.post(api_url, json=payload, headers=headers)
        
        if response.status_code == 401:
          self.auth.invalidate_token()
          raise UPSAPIError("Authentication failed", "AUTH_ERROR", 401)
        
        elif response.status_code != 200:
          error_text = response.text
          logger.error(f"UPS API returned status {response.status_code}: {error_text}")
          
          try:
            error_data = response.json()
            if "response" in error_data and "errors" in error_data["response"]:
              errors = error_data["response"]["errors"]
              if errors:
                error_msg = errors[0].get("message", "Unknown error")
                error_code = errors[0].get("code", "")
                
                raise UPSAPIError(f"UPS API error {error_code}: {error_msg}", error_code, response.status_code)
          except json.JSONDecodeError:
            logger.error(f"Failed to parse UPS error response as JSON: {error_text}")
          
          raise UPSAPIError(f"UPS API error: {response.status_code}", "API_ERROR", response.status_code)
        
        return response.json()
        
      except (httpx.TimeoutException, httpx.ConnectError) as e:
        if attempt == settings.UPS_MAX_RETRIES - 1:
          raise UPSAPIError("Connection timeout", "TIMEOUT_ERROR", 503)
        
        await asyncio.sleep(2 ** attempt)

  def _parse_rate_response(
    self,
    response_data: Dict[str, Any],
    origin_country: str,
    dest_country: str,
    context: str,
    requested_service: str = "all"
  ) -> List[RateQuote]:

    quotes = []

    try:
      rate_response = response_data.get("RateResponse", {})
      rated_shipments = rate_response.get("RatedShipment", [])

      if not rated_shipments:
        logger.warning(f"No RatedShipment found in response: {response_data}")
        return quotes

      if not isinstance(rated_shipments, list):
        rated_shipments = [rated_shipments]

      requested_service_code = None
      if requested_service != "all":
        requested_service_code = UPSConstants.SERVICES.get(requested_service)

      for shipment in rated_shipments:
        try:
          service = shipment.get("Service", {})
          service_code = service.get("Code", "")

          if requested_service_code and service_code != requested_service_code:
            continue

          service_name = UPSConstants.get_service_name_from_code(
            origin_country,
            dest_country,
            service_code
          )

          total_charges = shipment.get("TotalCharges", {})
          total_charge = float(total_charges.get("MonetaryValue", 0))
          currency_code = total_charges.get("CurrencyCode", "USD")

          delivery_days = None
          guaranteed = None

          if "GuaranteedDelivery" in shipment:
            guaranteed_info = shipment["GuaranteedDelivery"]
            if guaranteed_info.get("BusinessDaysInTransit"):
              delivery_days = int(guaranteed_info["BusinessDaysInTransit"])
              guaranteed = True

          quote = RateQuote(
            service_code=service_code,
            service_name=service_name,
            total_charge=total_charge,
            currency_code=currency_code,
            delivery_days=delivery_days,
            guaranteed=guaranteed,
            context=context,
            method_id=self._get_service_id(service_code, context)
          )

          quotes.append(quote)

        except Exception as e:
          logger.warning(f"Failed to parse shipment: {e}")
          continue

      return quotes

    except Exception as e:
      logger.error(f"Failed to parse UPS response: {e}")
      raise UPSAPIError("Failed to parse response", "PARSE_ERROR")
