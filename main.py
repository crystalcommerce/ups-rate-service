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
from core.logging import setup_logging, get_logger
from core.constants.ups import UPSConstants
from core.config import settings
from core.exceptions import UPSAPIError, USPSAPIError
setup_logging()  
logger = logging.getLogger(__name__)

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
