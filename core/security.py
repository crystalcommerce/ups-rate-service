from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.config import AUTH_TOKEN

security = HTTPBearer(auto_error=False)

async def require_api_key(
  credentials: HTTPAuthorizationCredentials = Depends(security)
):
  if not AUTH_TOKEN:
    return True

  if not credentials:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Missing authorization header"
    )

  if credentials.scheme.lower() != "bearer":
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Invalid authentication scheme"
    )

  if credentials.credentials != AUTH_TOKEN:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Invalid API key"
    )

  return True