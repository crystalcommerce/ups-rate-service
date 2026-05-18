import time
import requests
from core.config import settings

class FedexOauth:
  _access_token = None
  _expires_at = 0

  def __init__(self):
    self.base_url = settings.FEDEX_BASE_URL

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
    url = f"{self.base_url}/oauth/token"

    payload = {
      "grant_type": "client_credentials",
      "client_id": settings.FEDEX_CLIENT_ID,
      "client_secret": settings.FEDEX_CLIENT_SECRET
    }

    headers = {
      "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(url, data=payload, headers=headers)
    response.raise_for_status()

    data = response.json()

    self._access_token = data["access_token"]

    self._expires_at = time.time() + data["expires_in"] - 60

    return self._access_token
