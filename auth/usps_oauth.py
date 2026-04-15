import time
import requests

from core.config import settings
from core.exceptions import USPSAPIError


class USPSOauth:
  _cache = {}

  def __init__(self):
    self.url = settings.USPS_OAUTH_URL

  def get_token(self, scope: str = "addresses"):
    if self._is_token_valid(scope):
      return self._cache[scope]["access_token"]

    return self._generate_token(scope)

  def _is_token_valid(self, scope: str):
    token_data = self._cache.get(scope)

    return (
      token_data and
      token_data.get("access_token") and
      time.time() < token_data.get("expires_at", 0)
    )

  def _generate_token(self, scope: str):
    payload = {
      "grant_type": "client_credentials",
      "client_id": settings.USPS_CLIENT_ID,
      "client_secret": settings.USPS_CLIENT_SECRET,
      "scope": scope
    }

    headers = {
      "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(self.url, data=payload, headers=headers)

    if response.status_code != 200:
      raise USPSAPIError(
        f"OAuth failed: {response.status_code}",
        "AUTH_ERROR",
        response.status_code
      )

    data = response.json()

    if "access_token" not in data:
      raise USPSAPIError("Invalid OAuth response", "INVALID_RESPONSE")

    self._cache[scope] = {
      "access_token": data["access_token"],
      "expires_at": time.time() + int(data.get("expires_in", 28800)) - 60
    }

    return self._cache[scope]["access_token"]

  def invalidate_token(self, scope: str = None):
    if scope:
      self._cache.pop(scope, None)
    else:
      self._cache.clear()
