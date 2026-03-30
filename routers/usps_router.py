import time
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from schemas.ups import UPSRateRequest, RateResponse
from services.usps_service import USPSAddressService, USPSRateService
from core.exceptions.usps import USPSAPIError
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/usps", tags=["USPS"])


# USPS Address Validation Endpoint
@router.get("/addresses/validate", summary="Validate USPS address and get ZIP code")
async def validate_usps_address(
  streetAddress: str,
  city: str,
  state: str
) -> Dict[str, str]:
  """
  Validate a USPS address and return ZIP code information.
  Query parameters:
  - streetAddress: Street address
  - city: City name
  - state: Two-character state code

  Returns:
  - zipCode: 5-digit ZIP code
  - zipPlus4: 4-digit ZIP+4 code (if available)
  """
  request_id = f"usps_address_{int(time.time() * 1000)}"
  start_time = time.time()

  logger.info(f"USPS Address validation request {request_id}: {streetAddress}, {city}, {state}")

  try:
    async with USPSAddressService() as address_service:
      result = await address_service.validate_address(streetAddress, city, state)

    processing_time = (time.time() - start_time) * 1000
    logger.info(f"USPS Address validation {request_id} completed in {processing_time:.2f}ms")

    return result

  except USPSAPIError as e:
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"USPS Address validation {request_id} failed after {processing_time:.2f}ms: {str(e)}")
    raise HTTPException(
      status_code=e.status_code,
      detail={
        "error": e.error_code or "USPS_API_ERROR",
        "message": e.message,
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )
  except Exception as e:
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"USPS Address validation {request_id} unexpected error after {processing_time:.2f}ms: {str(e)}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail={
        "error": "INTERNAL_ERROR",
        "message": "An unexpected error occurred",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )


@router.post("/rates", response_model=RateResponse, summary="Get USPS rates (matching UPS format)")
async def get_usps_rates(request: UPSRateRequest) -> RateResponse:
  """
  Get USPS shipping rates - compatible with UPS microservice request format.

  Supports both domestic and international rates:
  - Domestic: Uses USPS domestic prices API
  - International: Uses USPS international prices API

  Returns rates in the same format as UPS microservice:
  {
      "quotes": [
          {
              "service_code": "PRIORITY_MAIL|SP",
              "service_name": "Priority Mail",
              "total_charge": 12.50,
              "context": "Domestic" or "International"
          }
      ]
  }
  """
  request_id = f"usps_rate_{int(time.time() * 1000)}"
  start_time = time.time()

  logger.info(f"USPS Rate request {request_id}: {request.sender_country} -> {request.country}, {request.weight} lbs")

  try:
    request.validate_required_fields()

    async with USPSRateService() as usps_service:
      quotes = await usps_service.get_rates(request, request_id)

    processing_time = (time.time() - start_time) * 1000
    is_domestic = request.country.upper() == "US" and request.sender_country.upper() == "US"
    context_str = "Domestic" if is_domestic else "International"

    response = RateResponse(
      quotes=quotes,
      request_id=request_id,
      timestamp=datetime.now(timezone.utc),
      processing_time_ms=processing_time,
      context=context_str
    )

    return response

  except ValueError as e:
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"USPS Rate request {request_id} validation failed after {processing_time:.2f}ms: {str(e)}")
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail={
        "error": "VALIDATION_ERROR",
        "message": str(e),
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )
  except USPSAPIError as e:
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"USPS Rate request {request_id} USPS API failed after {processing_time:.2f}ms: {str(e)}")
    raise HTTPException(
      status_code=e.status_code,
      detail={
        "error": e.error_code or "USPS_API_ERROR",
        "message": e.message,
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )
  except Exception as e:
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"USPS Rate request {request_id} unexpected error after {processing_time:.2f}ms: {str(e)}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail={
        "error": "INTERNAL_ERROR",
        "message": "An unexpected error occurred",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )