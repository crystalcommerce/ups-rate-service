"""Integration test for the FCMI wiring in USPSRateService.get_rates.

The USPS package-rates HTTP call and the OAuth token fetch are stubbed so the
test exercises only the rate-assembly flow: package quotes from the API plus
appended First-Class Mail International quotes from the static table.
"""
import services.usps_service as usps_service_mod
from schemas.rate_request import RateRequest


_FAKE_INTL_PACKAGE_RESPONSE = {
  "rateOptions": [
    {
      "totalBasePrice": 50.0,
      "rates": [
        {
          "mailClass": "PRIORITY_MAIL_INTERNATIONAL",
          "rateIndicator": "SP",
          "productName": "Priority Mail International",
        }
      ],
    }
  ]
}


def _stub_service(monkeypatch, package_response):
  svc = usps_service_mod.USPSRateService()
  monkeypatch.setattr(usps_service_mod.usps_oauth, "get_token", lambda scope=None: "fake-token")

  async def fake_request(token, payload, api_url, request_id, scope, log_label=""):
    return package_response

  monkeypatch.setattr(svc, "_make_usps_request", fake_request)
  return svc


async def test_get_rates_appends_fcmi_for_international(monkeypatch):
  svc = _stub_service(monkeypatch, _FAKE_INTL_PACKAGE_RESPONSE)
  request = RateRequest(sender_country="US", country="GB", sender_zip="90210", weight=0.1)

  quotes = await svc.get_rates(request, "test-req")

  # The package quote from the (stubbed) API is still present...
  assert any(q.service_code.startswith("PRIORITY_MAIL_INTERNATIONAL") for q in quotes)
  # ...and an FCMI quote has been appended from the static table.
  fcmi = [q for q in quotes if q.service_code == "FIRST-CLASS_MAIL"]
  assert len(fcmi) == 1
  assert fcmi[0].context == "International"
  assert "First-Class Mail International" in fcmi[0].service_name


async def test_get_rates_no_fcmi_for_overweight_international(monkeypatch):
  svc = _stub_service(monkeypatch, _FAKE_INTL_PACKAGE_RESPONSE)
  # 2 lb = 32 oz, over every FCMI limit — only the package quote should remain.
  request = RateRequest(sender_country="US", country="GB", sender_zip="90210", weight=2.0)

  quotes = await svc.get_rates(request, "test-req")

  assert all(q.service_code != "FIRST-CLASS_MAIL" for q in quotes)
