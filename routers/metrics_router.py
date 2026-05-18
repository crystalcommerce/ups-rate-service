from fastapi import APIRouter
from typing import Dict, Any

from core.config import settings
from core.constants.ups import UPSConstants

router = APIRouter(tags=["Metrics"])

@router.get("/metrics", summary="Service metrics")
async def get_metrics() -> Dict[str, Any]:
  return {
    "service": {
      "name": "UPS Rating Microservice",
      "version": "1.0.0",
      "environment": settings.UPS_ENVIRONMENT,
      "ruby_compatible": True,
      "supported_endpoints": [
        "/rates",
        "/parallel_rates", 
        "/ups_rate_request",
        "/services",
        "/service_name/{code}",
        "/constants",
        "/health",
        "/metrics"
      ]
    },
    "configuration": {
      "request_timeout": settings.UPS_REQUEST_TIMEOUT,
      "max_retries": settings.UPS_MAX_RETRIES,
      "api_version": UPSConstants.API_VERSION
    },
    "ruby_features": {
      "all_package_types": len(UPSConstants.PACKAGES),
      "all_service_types": len(UPSConstants.SERVICES),
      "regional_contexts": len(UPSConstants.SERVICE_CODES),
      "pickup_types": len(UPSConstants.PICKUP_TYPES),
      "customer_types": len(UPSConstants.CUSTOMER_TYPES)
    }
  }
