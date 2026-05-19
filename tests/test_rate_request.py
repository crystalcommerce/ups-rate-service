from schemas.rate_request import RateRequest


def _req(**overrides):
  base = dict(sender_country="US", country="US", weight=1.0)
  base.update(overrides)
  return RateRequest(**base)


def test_country_codes_are_upcased():
  req = _req(sender_country="us", country="ca")
  assert req.sender_country == "US"
  assert req.country == "CA"


def test_is_domestic_us_true_for_us_to_us():
  assert _req(sender_country="US", country="US").is_domestic_us() is True


def test_is_domestic_us_handles_lowercase():
  # Normalization makes the lowercase form behave like the uppercase form.
  assert _req(sender_country="us", country="us").is_domestic_us() is True


def test_is_domestic_us_false_for_international_destination():
  req = _req(sender_country="US", country="GB")
  assert req.is_domestic_us() is False
  assert req.is_international() is True


def test_is_international_true_when_sender_non_us():
  assert _req(sender_country="CA", country="US").is_international() is True
