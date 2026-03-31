from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

from core.constants.ups import UPSConstants
from core.logging import get_logger

logger = get_logger(__name__)


class Address(BaseModel):
  postal_code: str = Field(..., description="Postal/ZIP code")
  country_code: str = Field(..., description="Country code")

  city: Optional[str] = Field(None, description="City name")
  state_province_code: Optional[str] = Field(None, description="State/Province code")
  address_line: Optional[str] = Field(None, description="Street address")

  @validator('country_code')
  def validate_country_code(cls, v):
    return v.upper()


class Package(BaseModel):
  weight: float = Field(..., gt=0, description="Package weight")

  length: float = Field(0, ge=0, description="Length")
  width: float = Field(0, ge=0, description="Width")
  height: float = Field(0, ge=0, description="Height")

  packaging_type: str = Field("your_packaging", description="Package type")
  insured_value: float = Field(0, ge=0, description="Insured value")
  currency_code: str = Field("USD", description="Currency code")

  @validator('packaging_type')
  def validate_packaging_type(cls, v):
    if v not in UPSConstants.PACKAGES:
      logger.warning(f"Unknown packaging type: {v}, using default 'your_packaging'")
      return "your_packaging"
    return v

  @validator('weight')
  def validate_weight_minimum(cls, v):
    return max(1.0, float(v))


class UPSRateRequest(BaseModel):
  sender_country: str = Field(..., description="Sender country code")
  country: str = Field(..., description="Destination country code")
  weight: float = Field(..., gt=0, description="Package weight")

  zip: Optional[str] = Field(None, description="Destination ZIP code")
  sender_zip: Optional[str] = Field(None, description="Sender ZIP code")

  service: str = Field("all", description="Service type")
  package: str = Field("your_packaging", description="Package type")
  length: float = Field(0, ge=0, description="Package length")
  width: float = Field(0, ge=0, description="Package width")
  height: float = Field(0, ge=0, description="Package height")
  customer_type: str = Field("wholesale", description="Customer type")
  pickup_type: str = Field("daily_pickup", description="Pickup type")
  weight_units: str = Field("LBS", description="Weight units")
  measure_units: str = Field("IN", description="Dimension units")

  city: Optional[str] = Field(None, description="Destination city")
  state: Optional[str] = Field(None, description="Destination state")
  sender_city: Optional[str] = Field(None, description="Sender city")
  sender_state: Optional[str] = Field(None, description="Sender state")

  insured_value: float = Field(0, ge=0, description="Insured value")
  currency_code: str = Field("USD", description="Currency code")

  @validator('service')
  def validate_service(cls, v):
    if v not in UPSConstants.SERVICES:
      logger.warning(f"Unknown service: {v}, using default 'all'")
      return "all"
    return v

  @validator('package')
  def validate_package(cls, v):
    if v not in UPSConstants.PACKAGES:
      logger.warning(f"Unknown package type: {v}, using default 'your_packaging'")
      return "your_packaging"
    return v

  @validator('customer_type')
  def validate_customer_type(cls, v):
    if v not in UPSConstants.CUSTOMER_TYPES:
      logger.warning(f"Unknown customer type: {v}, using default 'wholesale'")
      return "wholesale"
    return v

  @validator('pickup_type')
  def validate_pickup_type(cls, v):
    if v not in UPSConstants.PICKUP_TYPES:
      logger.warning(f"Unknown pickup type: {v}, using default 'daily_pickup'")
      return "daily_pickup"
    return v

  def is_international(self) -> bool:
    return self.sender_country != "US" or self.country != "US"

  def is_domestic_us(self) -> bool:
    return self.sender_country == "US" and self.country == "US"

  def validate_required_fields(self):
    errors = []
    warnings = []

    for field in ["sender_country", "country", "weight"]:
      if not getattr(self, field, None):
        errors.append(f"Missing required field: {field}")

    if self.sender_country and len(self.sender_country) != 2:
      errors.append("sender_country must be a 2-character ISO country code")
    if self.country and len(self.country) != 2:
      errors.append("country must be a 2-character ISO country code")

    if self.is_domestic_us():
      if not self.zip or not self.sender_zip:
        errors.append("ZIP codes (zip and sender_zip) are required for US domestic shipments")

      if not self.city:
        warnings.append("Destination city is recommended for US domestic shipments for better accuracy")
      if not self.sender_city:
        warnings.append("Sender city is recommended for US domestic shipments for better accuracy")

      if not self.state:
        warnings.append("Destination state is recommended for US domestic shipments")
      if not self.sender_state:
        warnings.append("Sender state is recommended for US domestic shipments")

    elif self.is_international():
      if not self.zip:
        warnings.append("Destination postal code is strongly recommended for international shipments")
      if not self.sender_zip:
        warnings.append("Sender postal code is strongly recommended for international shipments")

      if not self.city:
        warnings.append("Destination city is recommended for international shipments")
      if not self.sender_city:
        warnings.append("Sender city is recommended for international shipments")

    if any([self.length, self.width, self.height]) and not all([
      self.length > 0,
      self.width > 0,
      self.height > 0
    ]):
      warnings.append("If providing dimensions, all three (length, width, height) should be greater than 0")

    for warning in warnings:
      logger.warning(warning)

    if errors:
      raise ValueError("; ".join(errors))


class RateQuote(BaseModel):
  service_code: str
  service_name: str
  total_charge: float
  currency_code: str
  delivery_days: Optional[int] = None
  guaranteed: Optional[bool] = None
  context: Optional[str] = None
  method_id: Optional[int] = None


class RateResponse(BaseModel):
  quotes: List[RateQuote]
  request_id: str
  timestamp: datetime
  processing_time_ms: float
  context: str
