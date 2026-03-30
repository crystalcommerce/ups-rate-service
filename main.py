import asyncio
import time
import base64
import httpx
import os
import logging
import json
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import     length: float = Field(0, ge=0, description="Package length")
HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
from routers import all_routers
from core.security import require_api_key
from core.logging import get_logger
from core.constants.ups import UPSConstants
from core.config import settings
from core.exceptions import UPSAPIError, USPSAPIError

logger = logging.getLogger(__name__)

# Set logging level with validation
try:
    log_level = settings.LOG_LEVEL.strip() if settings.LOG_LEVEL else "INFO"
    if hasattr(logging, log_level):
        logging.getLogger().setLevel(getattr(logging, log_level))
        logger.info(f"Logging level set to: {log_level}")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.warning(f"Invalid log level '{log_level}', defaulting to INFO")
except Exception as e:
    logging.getLogger().setLevel(logging.INFO)
    logger.error(f"Error setting log level: {e}, defaulting to INFO")

# Application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting UPS Rating Microservice")
    yield
    logger.info("Shutting down UPS Rating Microservice")

app = FastAPI(
    title="UPS Rating Microservice",
    description="UPS Rating API",
    version="1.0.0",
    lifespan=lifespan
)

for router in all_routers:
  app.include_router(
    router,
    dependencies=[Depends(require_api_key)]
  )

# CORS middleware
app.add_middleware(CORSMiddleware, 
                  allow_origins=["*"], 
                  allow_methods=["GET", "POST"], 
                  allow_headers=["*"])

# Security
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not settings.API_KEY:
        return True
    if not credentials or credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Exception handlers
@app.exception_handler(UPSAPIError)
async def ups_api_exception_handler(request: Request, exc: UPSAPIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code or "UPS_API_ERROR",
            "message": exc.message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(USPSAPIError)
async def usps_api_exception_handler(request: Request, exc: USPSAPIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code or "USPS_API_ERROR",
            "message": exc.message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# Main endpoint - Ruby compatible
@app.post("/rates", response_model=RateResponse, summary="Get UPS rates (Ruby UPSRateRequest compatible)")
async def get_ups_rates(
    request: UPSRateRequest,
    _: bool = Depends(verify_api_key)
) -> RateResponse:
    """
    Get UPS shipping rates - 100% compatible with Ruby UPSRateRequest class.
    
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

# USPS Address Validation Endpoint
@app.get("/usps/addresses/validate", summary="Validate USPS address and get ZIP code")
async def validate_usps_address(
    streetAddress: str,
    city: str,
    state: str,
    _: bool = Depends(verify_api_key)
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

@app.post("/usps/rates", response_model=RateResponse, summary="Get USPS rates (matching UPS format)")
async def get_usps_rates(
    request: UPSRateRequest,
    _: bool = Depends(verify_api_key)
) -> RateResponse:
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

# Health check and diagnostic endpoints
@app.get("/health", summary="Service health check")
async def health_check():
    """Comprehensive health check for the UPS microservice"""
    try:
        # Test OAuth token retrieval
        token_start = time.time()
        token = await token_manager.get_token()
        token_time = (time.time() - token_start) * 1000
        
        # Test UPS API connectivity
        connectivity_start = time.time()
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Just test if we can reach the UPS API endpoint
            try:
                response = await client.head(settings.RATE_URL.replace('/rating/v1/Rate', '/'), timeout=5.0)
                connectivity_status = "ok" if response.status_code in [200, 404, 405] else "degraded"
            except:
                connectivity_status = "failed"
        connectivity_time = (time.time() - connectivity_start) * 1000
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "checks": {
                "oauth_token": {
                    "status": "ok" if token else "failed",
                    "response_time_ms": round(token_time, 2)
                },
                "ups_api_connectivity": {
                    "status": connectivity_status,
                    "response_time_ms": round(connectivity_time, 2)
                }
            },
            "configuration": {
                "oauth_url": settings.UPS_OAUTH_URL,
                "rate_url": settings.UPS_RATE_URL,
                "timeout_seconds": settings.UPS_REQUEST_TIMEOUT,
                "max_retries": settings.UPS_MAX_RETRIES
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
        )

@app.get("/validate", summary="Validate request without making UPS API call")
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
        request = UPSRateRequest(
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
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid",
                "validation_passed": False,
                "errors": [str(e)]
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "validation_passed": False,
                "errors": [f"Validation error: {str(e)}"]
            }
        )







# Health and monitoring
# Duplicate health endpoint removed - using the comprehensive one above

@app.get("/metrics", summary="Service metrics")
async def get_metrics(_: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """
    Get service metrics and statistics.
    """
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



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )