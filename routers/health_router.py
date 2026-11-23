from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import time
import httpx

from core.config import settings
from auth.ups_oauth import UPSOauth

router = APIRouter(
  prefix="",
  tags=["Health"]
)


@router.get("/health")
async def health_check():
  try:
    # Health answers whether this process is serving, so it never fetches a token: a token call
    # that hangs would make every health check hang with it. It reports the cached UPS token as
    # "cached" or "not_cached"; rate requests fetch (and report) tokens themselves.
    token_start = time.time()
    token_status = "cached" if UPSOauth()._is_token_valid() else "not_cached"
    token_time = (time.time() - token_start) * 1000

    connectivity_start = time.time()

    async with httpx.AsyncClient(timeout=5.0) as client:
      try:
        response = await client.head(settings.UPS_RATE_URL.replace('/Rate', '/'))
        connectivity_status = (
          "ok" if response.status_code in [200, 404, 405] else "degraded"
        )
      except Exception:
        connectivity_status = "failed"

    connectivity_time = (time.time() - connectivity_start) * 1000

    is_healthy = connectivity_status in ["ok", "degraded"]

    response_body = {
      "status": "healthy" if is_healthy else "unhealthy",
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "checks": {
        "oauth_token": {
          "status": token_status,
          "response_time_ms": round(token_time, 2)
        },
        "ups_api_connectivity": {
          "status": connectivity_status,
          "response_time_ms": round(connectivity_time, 2)
        }
      }
    }

    if not is_healthy:
      return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response_body
      )

    return response_body

  except Exception as e:
    return JSONResponse(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      content={
        "status": "unhealthy",
        "error": str(e)
      }
    )
