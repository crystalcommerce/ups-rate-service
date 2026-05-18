from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


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
