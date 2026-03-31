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
    oauth = UPSOauth()

    token_start = time.time()
    token = oauth.get_token()
    token_time = (time.time() - token_start) * 1000
    token_status = "ok" if token else "failed"

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

    is_healthy = (
      token_status == "ok" and
      connectivity_status in ["ok", "degraded"]
    )

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