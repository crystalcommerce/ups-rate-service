"""Tests for FCMI pricing logic.

These assert structure and relative behavior (shape, context, monotonic
pricing) rather than exact dollar amounts, so they stay valid when the draft
price table is replaced with verified USPS Notice 123 numbers.
"""
from schemas.rate_request import RateRequest
from services.usps_fcmi import (
  _classify,
  _price_group,
  _to_oz,
  get_fcmi_quotes,
  _COUNTRY_GROUPS,
)


def _req(**overrides):
  base = dict(sender_country="US", country="GB", weight=0.1)  # 1.6 oz, international
  base.update(overrides)
  return RateRequest(**base)


# --- weight conversion ---

def test_to_oz_from_pounds():
  assert _to_oz(1.0, "LBS") == 16.0


def test_to_oz_from_ounces():
  assert _to_oz(2.5, "OZ") == 2.5


def test_to_oz_defaults_to_pounds():
  assert _to_oz(1.0, None) == 16.0


# --- shape classification ---

def test_classify_letter_by_weight_when_no_dimensions():
  assert _classify(2.0, (0, 0, 0)) == "letter"


def test_classify_large_envelope_for_heavier_weight():
  assert _classify(10.0, (0, 0, 0)) == "large_envelope"


def test_classify_none_when_over_every_limit():
  assert _classify(20.0, (0, 0, 0)) is None


def test_classify_postcard_with_postcard_dimensions():
  assert _classify(0.5, (6.0, 4.0, 0.01)) == "postcard"


def test_classify_letter_when_dims_exceed_postcard_but_fit_letter():
  assert _classify(0.5, (10.0, 5.0, 0.2)) == "letter"


def test_classify_large_envelope_when_dims_exceed_letter():
  assert _classify(0.5, (13.0, 10.0, 0.5)) == "large_envelope"


# --- price group lookup ---

def test_price_group_known_country_is_int():
  assert isinstance(_price_group("CA"), int)


def test_price_group_unknown_country_uses_default():
  assert _price_group("ZZ") == _COUNTRY_GROUPS["default_group"]


def test_price_group_is_case_insensitive():
  assert _price_group("gb") == _price_group("GB")


# --- end to end ---

def test_no_quote_for_domestic_shipment():
  req = RateRequest(sender_country="US", country="US", weight=0.1)
  assert get_fcmi_quotes(req) == []


def test_no_quote_for_non_us_origin():
  req = RateRequest(sender_country="CA", country="GB", weight=0.1)
  assert get_fcmi_quotes(req) == []


def test_no_quote_when_over_fcmi_max_weight():
  assert get_fcmi_quotes(_req(weight=2.0)) == []  # 32 oz


def test_letter_quote_shape():
  quotes = get_fcmi_quotes(_req(weight=0.1))  # 1.6 oz, no dims -> letter
  assert len(quotes) == 1
  quote = quotes[0]
  assert quote.service_code == "FIRST-CLASS_MAIL"
  assert quote.context == "International"
  assert "Letter" in quote.service_name
  assert quote.currency_code == "USD"
  assert quote.total_charge > 0


def test_large_envelope_quote_shape():
  quotes = get_fcmi_quotes(_req(weight=0.5))  # 8 oz -> large envelope
  assert len(quotes) == 1
  assert "Large Envelope" in quotes[0].service_name
  assert quotes[0].total_charge > 0


def test_postcard_quote_shape():
  quotes = get_fcmi_quotes(_req(weight=0.03, length=6.0, width=4.0, height=0.01))
  assert len(quotes) == 1
  assert "Postcard" in quotes[0].service_name


def test_heavier_letter_costs_at_least_as_much_as_lighter():
  light = get_fcmi_quotes(_req(weight=0.03))[0].total_charge  # ~0.48 oz
  heavy = get_fcmi_quotes(_req(weight=0.2))[0].total_charge   # ~3.2 oz
  assert heavy >= light
