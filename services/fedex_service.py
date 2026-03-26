import requests
import time

from auth.fedex_oauth import FedexOauth
from core.config import FEDEX_BASE_URL, FEDEX_ACCOUNT_NUMBER

class FedexService:
  def __init__(self):
    self.auth = FedexOauth()

  def get_rates(self, payload: dict):
    url = f"{FEDEX_BASE_URL}/rate/v1/rates/quotes"

    weight = payload['requestedShipment']["requestedPackageLineItems"][0]["weight"]
    weight["value"] = max(weight["value"], 1)

    payload = {
      **payload,
      "accountNumber": {
        "value": FEDEX_ACCOUNT_NUMBER
      }
    }

    return self._make_request_with_retry(url, payload)

  def _make_request_with_retry(self, url, payload, retries=3):
    attempt = 0

    while attempt < retries:
      token = self.auth.get_token()

      headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
      }

      try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code < 400:
          return response.json()

        if response.status_code == 401:
          self.auth._access_token = None
          attempt += 1
          continue

        if response.status_code >= 500:
          attempt += 1
          time.sleep(2 ** attempt)
          continue

        raise Exception({
          "status_code": response.status_code,
          "response": response.json()
        })

      except requests.exceptions.RequestException as e:
        attempt += 1
        time.sleep(2 ** attempt)

    raise Exception("FedEx API failed after retries")
