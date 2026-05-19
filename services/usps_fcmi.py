"""First-Class Mail International (FCMI) rate pricing.

USPS exposes no API for international letter/postcard/flat rates: the modern
developers.usps.com APIs cover international *packages* only, and the legacy
Web Tools `IntlRateV2` API that rated letters was retired 2026-01-25. FCMI
rates are therefore computed here from USPS-published static price tables.

Data files (refresh at each USPS price change, ~annually):
  core/data/usps_fcmi_prices.json          - prices by shape and weight tier
  core/data/usps_country_price_groups.json - destination country -> price group
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

from schemas.rate_request import RateRequest
from schemas.rate_response import RateQuote
from core.logging import get_logger

logger = get_logger(__name__)

# USPS mailpiece dimension limits, in inches, largest-first (length, height, thickness).
# Physical mailpiece standards (USPS DMM/IMM) — far more stable than the price tables.
LETTER_MAX_DIMS_IN = (11.5, 6.125, 0.25)
POSTCARD_MAX_DIMS_IN = (6.0, 4.25, 0.016)

_DATA_DIR = Path(__file__).resolve().parent.parent / "core" / "data"


def _load(filename: str) -> dict:
  with open(_DATA_DIR / filename, encoding="utf-8") as fh:
    return json.load(fh)


_PRICES = _load("usps_fcmi_prices.json")
_COUNTRY_GROUPS = _load("usps_country_price_groups.json")


def _to_oz(weight: float, units: Optional[str]) -> float:
  """Convert a request weight to ounces. Defaults to pounds (core/admin sends LBS)."""
  return float(weight) if (units or "LBS").upper() == "OZ" else float(weight) * 16.0


def _dims_in(request: RateRequest) -> Tuple[float, float, float]:
  """Return (length, height, thickness) in inches, largest-first; (0, 0, 0) if absent."""
  raw = [request.length or 0, request.width or 0, request.height or 0]
  if (request.measure_units or "IN").upper() == "CM":
    raw = [d / 2.54 for d in raw]
  ordered = sorted((float(d) for d in raw), reverse=True)
  return ordered[0], ordered[1], ordered[2]


def _within(dims: Tuple[float, float, float], limits: Tuple[float, float, float]) -> bool:
  return all(d <= lim for d, lim in zip(dims, limits))


def _classify(weight_oz: float, dims: Tuple[float, float, float]) -> Optional[str]:
  """Pick the FCMI shape for a mailpiece, or None if it exceeds every FCMI limit.

  With no dimensions we fall back to weight alone (letter if light enough, else
  large envelope) — mirroring how the domestic letter-rates path defaults dims.
  """
  has_dims = all(d > 0 for d in dims)
  max_oz = _PRICES["max_weight_oz"]

  if has_dims and weight_oz <= max_oz["postcard"] and _within(dims, POSTCARD_MAX_DIMS_IN):
    return "postcard"
  if weight_oz <= max_oz["letter"] and (not has_dims or _within(dims, LETTER_MAX_DIMS_IN)):
    return "letter"
  if weight_oz <= max_oz["large_envelope"]:
    return "large_envelope"
  return None


def _price_group(country_code: str) -> int:
  return _COUNTRY_GROUPS["groups"].get(
    (country_code or "").upper(), _COUNTRY_GROUPS["default_group"]
  )


def _tier_price(shape_data: dict, weight_oz: float, price_group: int) -> Optional[float]:
  """Look up the price for a shape at a given weight and destination price group."""
  for tier in shape_data["tiers"]:  # ordered ascending by max_oz
    if weight_oz <= tier["max_oz"]:
      if shape_data["pricing"] == "worldwide":
        return tier["price"]
      return tier["prices"].get(str(price_group))
  return None


def get_fcmi_quotes(request: RateRequest) -> List[RateQuote]:
  """Return First-Class Mail International quote(s) for a US-origin outbound shipment.

  Returns an empty list for domestic shipments, non-US origins, or mailpieces
  that exceed every FCMI weight limit.
  """
  if request.is_domestic_us() or request.sender_country != "US":
    return []

  weight_oz = _to_oz(request.weight, request.weight_units)
  dims = _dims_in(request)
  shape = _classify(weight_oz, dims)
  if shape is None:
    logger.info(f"[FCMI] {weight_oz:.2f}oz exceeds every FCMI limit; no FCMI quote")
    return []

  group = _price_group(request.country)
  shape_data = _PRICES[shape]
  price = _tier_price(shape_data, weight_oz, group)
  if price is None:
    logger.warning(
      f"[FCMI] no {shape} price for {weight_oz:.2f}oz, "
      f"price group {group} ({request.country})"
    )
    return []

  logger.info(
    f"[FCMI] {shape} to {request.country} (group {group}), "
    f"{weight_oz:.2f}oz -> ${price:.2f}"
  )
  return [RateQuote(
    service_code="FIRST-CLASS_MAIL",
    service_name=shape_data["service_name"],
    total_charge=float(price),
    currency_code="USD",
    context="International",
  )]
