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
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
from routers import public_routers, secured_routers
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

for router in public_routers:
  app.include_router(router)

for router in secured_routers:
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







if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )