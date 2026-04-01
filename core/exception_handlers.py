from fastapi import Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from core.exceptions import UPSAPIError, USPSAPIError
from core.logging import get_logger

logger = get_logger(__name__)

async def ups_api_exception_handler(request: Request, exc: UPSAPIError):
  logger.error(f"UPSAPIError: {exc.message}")
  return JSONResponse(
    status_code=exc.status_code,
    content={
      "error": exc.error_code or "UPS_API_ERROR",
      "message": exc.message,
      "timestamp": datetime.now(timezone.utc).isoformat()
    }
  )

async def usps_api_exception_handler(request: Request, exc: USPSAPIError):
  logger.error(f"USPSAPIError: {exc.message}")
  return JSONResponse(
    status_code=exc.status_code,
    content={
      "error": exc.error_code or "USPS_API_ERROR",
      "message": exc.message,
      "timestamp": datetime.now(timezone.utc).isoformat()
    }
  )
