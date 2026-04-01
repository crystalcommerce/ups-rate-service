import time
import base64
import requests

from core.config import settings
from core.exceptions import UPSAPIError


class UPSOauth:
  _access_token = None
  _expires_at = 0

  def __init__(self):
    self.url = settings.UPS_OAUTH_URL

  def get_token(self):
    if self._is_token_valid():
      return self._access_token

    return self._generate_token()

  def _is_token_valid(self):
    return (
      self._access_token is not None and
      time.time() < self._expires_at
    )

  def _generate_token(self):
    credentials = f"{settings.UPS_CLIENT_ID}:{settings.UPS_CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
      "Content-Type": "application/x-www-form-urlencoded",
      "Authorization": f"Basic {encoded_credentials}"
    }

    payload = {
      "grant_type": "client_credentials"
    }

    response = requests.post(self.url, data=payload, headers=headers)

    if response.status_code != 200:
      raise UPSAPIError(
        f"OAuth failed: {response.status_code}",
        status_code=response.status_code
      )

    data = response.json()

    if "access_token" not in data:
      raise UPSAPIError("Invalid OAuth response")

    self._access_token = data["access_token"]
    self._expires_at = time.time() + int(data.get("expires_in", 3600)) - 60

    return self._access_token

  def invalidate_token(self):
    self._access_token = None
    self._expires_at = 0
