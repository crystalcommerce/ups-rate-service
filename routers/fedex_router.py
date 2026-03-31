import time
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status

from services.fedex_service import FedexService
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/fedex", tags=["FedEx"])

service = FedexService()


@router.post("/rates", summary="Get FedEx shipping rates")
async def get_fedex_rates(payload: Dict[str, Any]) -> Dict[str, Any]:
  """
  Get FedEx shipping rates.

  This endpoint accepts a request payload containing shipment details
  and returns FedEx shipping rates in a structured format.

  Returns:
  {
    "service_code": "FEDEX_GROUND",
    "service_name": "FedEx Ground",
    "total_charge": 12.50,
    "currency": "USD"
  }
  """
  request_id = f"fedex_rate_{int(time.time() * 1000)}"
  start_time = time.time()

  logger.info(f"FedEx Rate request {request_id}: payload keys = {list(payload.keys())}")

  try:
    result = service.get_rates(payload)

    processing_time = (time.time() - start_time) * 1000
    logger.info(f"FedEx Rate request {request_id} completed in {processing_time:.2f}ms")

    return result

  except ValueError as e:
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"FedEx Rate request {request_id} validation failed after {processing_time:.2f}ms: {str(e)}")
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail={
        "error": "VALIDATION_ERROR",
        "message": str(e),
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )
  except Exception as e:
    processing_time = (time.time() - start_time) * 1000
    logger.error(f"FedEx Rate request {request_id} unexpected error after {processing_time:.2f}ms: {str(e)}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail={
        "error": "INTERNAL_ERROR",
        "message": "An unexpected error occurred",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
      }
    )
