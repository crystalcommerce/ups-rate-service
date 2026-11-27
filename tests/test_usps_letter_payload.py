from schemas.rate_request import RateRequest
from services.usps_service import USPSRateService


def _req(**overrides):
  base = dict(sender_country="US", country="US", weight=0.1)  # 0.1 lb = 1.6 oz
  base.update(overrides)
  return RateRequest(**base)


def test_letter_payload_none_for_international():
  svc = USPSRateService()
  assert svc._build_letter_payload(_req(country="GB")) is None


def test_letter_payload_none_when_over_letter_weight():
  svc = USPSRateService()
  # 0.25 lb = 4 oz, over the 3.5 oz First-Class Mail letter limit.
  assert svc._build_letter_payload(_req(weight=0.25)) is None


def test_letter_payload_uses_defaults_without_dimensions():
  svc = USPSRateService()
  payload = svc._build_letter_payload(_req(weight=0.1))
  assert payload is not None
  assert payload["processingCategory"] == "LETTERS"
  assert payload["length"] == 6.0
  assert payload["height"] == 4.0
  assert payload["thickness"] == 0.02


def test_letter_payload_sorts_dimensions_ascending():
  svc = USPSRateService()
  payload = svc._build_letter_payload(
    _req(weight=0.1, length=11.0, width=0.02, height=5.0)
  )
  # Dimensions are sorted ascending into thickness <= height <= length.
  assert payload["thickness"] == 0.02
  assert payload["height"] == 5.0
  assert payload["length"] == 11.0


def test_rate_payload_international_is_at_least_the_smallest_parcel():
  # Below 6 x 4 in USPS prices no First-Class Package International (Category One, 2026-11-27).
  svc = USPSRateService()
  payload = svc._build_rate_payload(_req(sender_zip="97301", country="CA", zip="M5V 2T6", weight=0.5))
  assert (payload["length"], payload["width"], payload["height"]) == (6.0, 4.0, 1.0)


def test_rate_payload_international_keeps_a_larger_parcel():
  svc = USPSRateService()
  payload = svc._build_rate_payload(
    _req(sender_zip="97301", country="CA", zip="M5V 2T6", weight=2.0, length=12.0, width=9.0, height=3.0)
  )
  assert (payload["length"], payload["width"], payload["height"]) == (12.0, 9.0, 3.0)


def test_rate_payload_domestic_unchanged():
  svc = USPSRateService()
  payload = svc._build_rate_payload(_req(sender_zip="97301", zip="10001", weight=0.5))
  assert (payload["length"], payload["width"], payload["height"]) == (1.0, 1.0, 1.0)
