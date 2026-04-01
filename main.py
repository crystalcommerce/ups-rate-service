import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from routers import public_routers, secured_routers
from core.logging import setup_logging, get_logger
from core.security import require_api_key
from core.exception_handlers import ups_api_exception_handler, usps_api_exception_handler
from core.exceptions import UPSAPIError, USPSAPIError

setup_logging()
logger = get_logger(__name__)

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

# Include routers
for r in public_routers: app.include_router(r)
for r in secured_routers: app.include_router(r, dependencies=[Depends(require_api_key)])

# CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Exception handlers
app.add_exception_handler(UPSAPIError, ups_api_exception_handler)
app.add_exception_handler(USPSAPIError, usps_api_exception_handler)
