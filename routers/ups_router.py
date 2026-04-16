from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone
import time
import logging

from schemas.rate_request import RateRequest
from schemas.rate_response import RateResponse
from services.ups_service import UPSRatingService
from core.constants.ups import UPSConstants
from core.exceptions.ups import UPSAPIError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ups", tags=["UPS"])


# Main endpoint - Ruby compatible
@router.post("/rates", response_model=RateResponse, summary="Get UPS rates (Ruby RateRequest compatible)")
async def get_ups_rates(request: RateRequest) -> RateResponse:
  """
  Get UPS shipping rates - 100% compatible with Ruby RateRequest class.
  
  Supports all the same parameters and logic as the original Ruby implementation:
  - All service types (next_day, 2_day, ground, worldwide_express, etc.)
  - All package types (ups_envelope, your_packaging, ups_tube, etc.) 
  - All pickup types (daily_pickup, customer_counter, one_time_pickup, etc.)
  - All customer types (wholesale, occasional, retail)
  - Regional service contexts (US Domestic, US Origin, Canada Origin, etc.)
  - Insurance and special services
  - Proper weight minimums (1.0 lb minimum)
  - International vs domestic logic
  - Rate shopping (multiple quotes) by default
  """

  request_id = f"ups_rate_{int(time.time() * 1000)}"
  start_time = time.time()

  # Store request_id in local variable to ensure scope availability
  local_request_id = request_id

  logger.info(f"UPS Rate request {local_request_id}: {request.sender_country} -> {request.country}, {request.weight} {request.weight_units}, service: {request.service}")

  try:
    async with UPSRatingService() as ups_service:
      quotes = await ups_service.get_rates(request, local_request_id, shop_rates=True)

    processing_time = (time.time() - start_time) * 1000
    context = UPSConstants.get_context(request.sender_country, request.country)

    response = RateResponse(
      quotes=quotes,
      request_id=local_request_id,
      timestamp=datetime.now(timezone.utc),
      processing_time_ms=processing_time,
      context=context
    )

    return response

  except ValueError as e:
    # Validation errors should return 400 Bad Request
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"Rate request {local_request_id} validation failed after {processing_time:.2f}ms: {str(e)}")

    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail={
        "error": "VALIDATION_ERROR",
        "message": str(e),
        "request_id": local_request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )

  except UPSAPIError as e:
    # UPS API errors should use their specific status codes
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"Rate request {local_request_id} UPS API failed after {processing_time:.2f}ms: {str(e)}")

    raise HTTPException(
      status_code=e.status_code,
      detail={
        "error": e.error_code or "UPS_API_ERROR",
        "message": e.message,
        "request_id": local_request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )

  except Exception as e:
    # Unexpected errors should return 500
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"Rate request {local_request_id} unexpected error after {processing_time:.2f}ms: {str(e)}")

    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail={
        "error": "INTERNAL_ERROR",
        "message": "An unexpected error occurred",
        "request_id": local_request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )


@router.get("/validate", summary="Validate request without making UPS API call")
async def validate_request(
  sender_country: str,
  country: str,
  weight: float,
  zip: str = None,
  sender_zip: str = None,
  city: str = None,
  sender_city: str = None,
  state: str = None,
  sender_state: str = None,
  service: str = "all",
  package: str = "your_packaging"
):
  """Validate a shipping request without making an actual UPS API call"""

  try:
    # Create request object
    request = RateRequest(
      sender_country=sender_country,
      country=country,
      weight=weight,
      zip=zip,
      sender_zip=sender_zip,
      city=city,
      sender_city=sender_city,
      state=state,
      sender_state=sender_state,
      service=service,
      package=package
    )

    # Validate the request
    request.validate_required_fields()

    # Determine shipment context
    context = UPSConstants.get_context(sender_country, country)

    return {
      "status": "valid",
      "context": context,
      "shipment_type": "domestic_us" if request.is_domestic_us() else "international",
      "validation_passed": True,
      "recommendations": {
        "service_available": service in UPSConstants.SERVICES,
        "package_type_valid": package in UPSConstants.PACKAGES,
        "weight_adjusted": max(0.1, weight) != weight
      }
    }

  except ValueError as e:
    raise HTTPException(
      status_code=400,
      detail={
        "status": "invalid",
        "validation_passed": False,
        "errors": [str(e)]
      }
    )

  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail={
        "status": "error",
        "validation_passed": False,
        "errors": [f"Validation error: {str(e)}"]
      }
    )
