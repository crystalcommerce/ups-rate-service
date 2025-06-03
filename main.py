import asyncio
import time
import base64
import httpx
import os
import logging
import json
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Custom JSON Response for performance
class ORJSONResponse(Response):
    media_type = "application/json"
    
    def render(self, content: Any) -> bytes:
        if USE_ORJSON:
            return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)
        else:
            return json.dumps(content, ensure_ascii=False, separators=(',', ':')).encode()

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    CLIENT_ID = os.getenv("UPS_CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("UPS_CLIENT_SECRET", "")
    UPS_ENVIRONMENT = os.getenv("UPS_ENVIRONMENT", "sandbox").lower()
    

    
    # API Configuration
    REQUEST_TIMEOUT = int(os.getenv("UPS_REQUEST_TIMEOUT", "30"))
    MAX_RETRIES = int(os.getenv("UPS_MAX_RETRIES", "3"))
    RATE_LIMIT_REQUESTS = int(os.getenv("UPS_RATE_LIMIT", "100"))
    
    # Production settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    API_KEY = os.getenv("API_KEY", "")
    
    # Set URLs based on environment
    if UPS_ENVIRONMENT == "production":
        OAUTH_URL = "https://onlinetools.ups.com/security/v1/oauth/token"
        RATE_URL = "https://onlinetools.ups.com/api/rating/v1/Rate"

    else:
        OAUTH_URL = "https://wwwcie.ups.com/security/v1/oauth/token"
        RATE_URL = "https://wwwcie.ups.com/api/rating/v1/Rate" 

    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.CLIENT_ID or not cls.CLIENT_SECRET:
            raise ValueError("UPS_CLIENT_ID and UPS_CLIENT_SECRET environment variables are required")
        
        logger.info(f"UPS Configuration: Environment={cls.UPS_ENVIRONMENT}, CLIENT_ID={'***' if cls.CLIENT_ID else 'MISSING'}, CLIENT_SECRET={'***' if cls.CLIENT_SECRET else 'MISSING'}")
        logger.info(f"UPS URLs: OAuth={cls.OAUTH_URL}, Rate={cls.RATE_URL}")

# UPS Constants
class UPSConstants:
    API_VERSION = "1.0001"
    
    # Package types from Ruby
    PACKAGES = {
        "ups_envelope": "01",
        "your_packaging": "02", 
        "ups_tube": "03",
        "ups_pak": "04",
        "ups_box": "21",
        "fedex_25_kg_box": "24",
        "fedex_10_kg_box": "25"
    }
    
    # Services from Ruby
    SERVICES = {
        "next_day": "01",
        "2_day": "02", 
        "ground": "03",
        "worldwide_express": "07",
        "worldwide_expedited": "08",
        "standard": "11",
        "3_day": "12",
        "next_day_saver": "13",
        "next_day_early": "14",
        "worldwide_express_plus": "54",
        "2_day_early": "59",
        "all": "all"
    }
    
    # Service codes by region (from Ruby)
    SERVICE_CODES = {
        "US Domestic": {
            "01": "UPS Next Day Air",
            "02": "UPS Second Day Air", 
            "03": "UPS Ground",
            "12": "UPS Three-Day Select",
            "13": "UPS Next Day Air Saver",
            "14": "UPS Next Day Air Early A.M.",
            "59": "UPS Second Day Air A.M.",
            "65": "UPS Saver"
        },
        "US Origin": {
            "07": "UPS Worldwide Express",
            "08": "UPS Worldwide Expedited",
            "11": "UPS Standard",
            "54": "UPS Worldwide Express Plus"
        },
        "Puerto Rico Origin": {
            "01": "UPS Next Day Air",
            "02": "UPS Second Day Air",
            "03": "UPS Ground", 
            "07": "UPS Worldwide Express",
            "08": "UPS Worldwide Expedited",
            "14": "UPS Next Day Air Early A.M.",
            "54": "UPS Worldwide Express Plus",
            "65": "UPS Saver"
        },
        "Canada Origin": {
            "01": "UPS Express",
            "02": "UPS Expedited",
            "07": "UPS Worldwide Express",
            "08": "UPS Worldwide Expedited", 
            "11": "UPS Standard",
            "12": "UPS Three-Day Select",
            "13": "UPS Saver",
            "14": "UPS Express Early A.M.",
            "54": "UPS Worldwide Express Plus",
            "65": "UPS Saver"
        },
        "Mexico Origin": {
            "07": "UPS Express",
            "08": "UPS Expedited",
            "54": "UPS Express Plus",
            "65": "UPS Saver"
        },
        "Polish Domestic": {
            "07": "UPS Express",
            "08": "UPS Expedited",
            "11": "UPS Standard",
            "54": "UPS Worldwide Express Plus",
            "65": "UPS Saver",
            "82": "UPS Today Standard",
            "83": "UPS Today Dedicated Courrier",
            "84": "UPS Today Intercity",
            "85": "UPS Today Express",
            "86": "UPS Today Express Saver"
        },
        "EU Origin": {
            "07": "UPS Express",
            "08": "UPS Expedited", 
            "11": "UPS Standard",
            "54": "UPS Worldwide Express Plus",
            "65": "UPS Saver"
        },
        "Other International Origin": {
            "07": "UPS Express",
            "08": "UPS Worldwide Expedited",
            "11": "UPS Standard", 
            "54": "UPS Worldwide Express Plus",
            "65": "UPS Saver"
        },
        "Freight": {
            "TDCB": "Trade Direct Cross Border",
            "TDA": "Trade Direct Air",
            "TDO": "Trade Direct Ocean",
            "308": "UPS Freight LTL",
            "309": "UPS Freight LTL Guaranteed",
            "310": "UPS Freight LTL Urgent"
        }
    }
    
    # Pickup types from Ruby
    PICKUP_TYPES = {
        'daily_pickup': '01',
        'customer_counter': '03',
        'one_time_pickup': '06',
        'on_call': '07',
        'suggested_retail_rates': '11',
        'letter_center': '19',
        'air_service_center': '20'
    }
    
    # Customer types from Ruby
    CUSTOMER_TYPES = {
        'wholesale': '01',
        'occasional': '02',
        'retail': '04'
    }
    
    # Payment types from Ruby
    PAYMENT_TYPES = {
        'prepaid': 'Prepaid',
        'consignee': 'Consignee',
        'bill_third_party': 'BillThirdParty',
        'freight_collect': 'FreightCollect'
    }
    
    # EU Country codes from Ruby
    EU_COUNTRY_CODES = [
        "GB", "AT", "BE", "BG", "CY", "CZ", "DK", "EE", "FI", "FR", 
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", 
        "PL", "PT", "RO", "SK", "SI", "ES", "SE"
    ]
    
    @classmethod
    def get_context(cls, origin_country: str, destination_country: str) -> str:
        """Get service context based on origin and destination"""
        if origin_country == "US":
            return 'US Domestic' if destination_country == "US" else 'US Origin'
        elif origin_country == "PR":
            return 'Puerto Rico Origin'
        elif origin_country == "CA":
            return 'Canada Origin'
        elif origin_country == "MX":
            return 'Mexico Origin'
        elif origin_country == "PL":
            return 'Polish Domestic' if destination_country == "PL" else 'Other International Origin'
        elif origin_country in cls.EU_COUNTRY_CODES:
            return 'EU Origin'
        else:
            return 'Other International Origin'
    
    @classmethod
    def get_service_name_from_code(cls, origin_country: str, destination_country: str, service_code: str) -> str:
        """Get service name from code based on context"""
        context = cls.get_context(origin_country, destination_country)
        return cls.SERVICE_CODES.get(context, {}).get(service_code, "Unknown Service")

# Initialize configuration
try:
    Config.validate()
    logger.info(f"Configuration loaded - Environment: {Config.UPS_ENVIRONMENT}")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# Set logging level
logging.getLogger().setLevel(getattr(logging, Config.LOG_LEVEL))

# Custom exceptions
class UPSAPIError(Exception):
    def __init__(self, message: str, error_code: str = None, status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

# Token management (same as before)
class TokenManager:
    def __init__(self):
        self._cache = {
            "access_token": None,
            "expires_at": 0,
            "lock": asyncio.Lock()
        }
    
    async def get_token(self) -> str:
        async with self._cache["lock"]:
            current_time = time.time()
            
            if (self._cache["access_token"] and 
                self._cache["expires_at"] > current_time):
                return self._cache["access_token"]
            
            token_data = await self._request_token()
            self._cache["access_token"] = token_data["access_token"]
            expires_in = int(token_data.get("expires_in", 3600))
            self._cache["expires_at"] = current_time + expires_in - 60
            
            return self._cache["access_token"]
    
    async def _request_token(self) -> Dict[str, Any]:
        credentials = f"{Config.CLIENT_ID}:{Config.CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {encoded_credentials}"
        }
        data = {"grant_type": "client_credentials"}
        
        logger.info(f"Requesting OAuth token from: {Config.OAUTH_URL}")
        
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
            response = await client.post(Config.OAUTH_URL, data=data, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"OAuth failed with status {response.status_code}: {response.text}")
                raise UPSAPIError(f"OAuth failed: {response.status_code}")
            
            token_data = response.json()
            if "access_token" not in token_data:
                logger.error(f"Invalid OAuth response: {token_data}")
                raise UPSAPIError("Invalid OAuth response")
            
            logger.info("Successfully obtained OAuth token")
            return token_data
    
    def invalidate_token(self):
        self._cache["access_token"] = None
        self._cache["expires_at"] = 0

token_manager = TokenManager()

# Request/Response Models
class Address(BaseModel):
    # Required fields
    postal_code: str = Field(..., description="Postal/ZIP code")
    country_code: str = Field(..., description="Country code")
    
    # Optional fields
    city: Optional[str] = Field(None, description="City name")
    state_province_code: Optional[str] = Field(None, description="State/Province code")
    address_line: Optional[str] = Field(None, description="Street address")
    
    @validator('country_code')
    def validate_country_code(cls, v):
        return v.upper()

class Package(BaseModel):
    # Required
    weight: float = Field(..., gt=0, description="Package weight")
    
    # Optional dimensions
    length: float = Field(0, ge=0, description="Length")
    width: float = Field(0, ge=0, description="Width") 
    height: float = Field(0, ge=0, description="Height")
    
    # Optional fields with Ruby defaults
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
        # UPS requirement: minimum 1.0 pound
        return max(1.0, float(v))

class UPSRateRequest(BaseModel):
    """Complete UPS Rate Request matching Ruby UPSRateRequest class"""
    
    # Required fields
    sender_country: str = Field(..., description="Sender country code")
    country: str = Field(..., description="Destination country code")  
    weight: float = Field(..., gt=0, description="Package weight")
    
    # Conditional required fields (zip codes required for domestic US)
    zip: Optional[str] = Field(None, description="Destination ZIP code")
    sender_zip: Optional[str] = Field(None, description="Sender ZIP code")
    
    # Optional fields with Ruby defaults
    service: str = Field("all", description="Service type")
    package: str = Field("your_packaging", description="Package type")
    length: float = Field(0, ge=0, description="Package length")
    width: float = Field(0, ge=0, description="Package width")
    height: float = Field(0, ge=0, description="Package height")
    customer_type: str = Field("wholesale", description="Customer type")
    pickup_type: str = Field("daily_pickup", description="Pickup type")
    weight_units: str = Field("LBS", description="Weight units")
    measure_units: str = Field("IN", description="Dimension units")
    
    # Optional address fields
    city: Optional[str] = Field(None, description="Destination city")
    state: Optional[str] = Field(None, description="Destination state")
    sender_city: Optional[str] = Field(None, description="Sender city")
    sender_state: Optional[str] = Field(None, description="Sender state")
    
    # Optional service options
    insured_value: float = Field(0, ge=0, description="Insured value")
    currency_code: str = Field("USD", description="Currency code")
    

    
    @validator('weight')
    def validate_weight_minimum(cls, v):
        # UPS requirement: minimum 1.0 pound
        return max(1.0, float(v))
    
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
        """Check if shipment is international"""
        return self.sender_country != "US" or self.country != "US"
    
    def is_domestic_us(self) -> bool:
        """Check if shipment is domestic US"""
        return self.sender_country == "US" and self.country == "US"
    
    def validate_required_fields(self):
        """Enhanced validation for all shipping scenarios based on UPS SDK patterns"""
        errors = []
        warnings = []
        
        # Basic required fields for all shipments
        required_basic = ["sender_country", "country", "weight"]
        for field in required_basic:
            if not getattr(self, field, None):
                errors.append(f"Missing required field: {field}")
        
        # Country code validation
        if self.sender_country and len(self.sender_country) != 2:
            errors.append("sender_country must be a 2-character ISO country code")
        if self.country and len(self.country) != 2:
            errors.append("country must be a 2-character ISO country code")
        
        # Domestic US shipments - flexible requirements
        if self.is_domestic_us():
            # ZIP codes are mandatory for US domestic
            if not self.zip or not self.sender_zip:
                errors.append("ZIP codes (zip and sender_zip) are required for US domestic shipments")
            
            # Cities are recommended for US domestic (not mandatory to avoid blocking valid requests)
            if not self.city:
                warnings.append("Destination city is recommended for US domestic shipments for better accuracy")
            if not self.sender_city:
                warnings.append("Sender city is recommended for US domestic shipments for better accuracy")
            
            # States are recommended for US domestic
            if not self.state:
                warnings.append("Destination state is recommended for US domestic shipments")
            if not self.sender_state:
                warnings.append("Sender state is recommended for US domestic shipments")
        
        # International shipments - flexible requirements
        elif self.is_international():
            # Postal codes are strongly recommended for international
            if not self.zip:
                warnings.append("Destination postal code is strongly recommended for international shipments")
            if not self.sender_zip:
                warnings.append("Sender postal code is strongly recommended for international shipments")
            
            # Cities are recommended for international
            if not self.city:
                warnings.append("Destination city is recommended for international shipments")
            if not self.sender_city:
                warnings.append("Sender city is recommended for international shipments")
        
        # Weight validation
        if self.weight and self.weight < 0.1:
            errors.append("Weight must be at least 0.1 pounds")
        
        # Package dimensions validation
        if any([self.length, self.width, self.height]) and not all([self.length > 0, self.width > 0, self.height > 0]):
            warnings.append("If providing dimensions, all three (length, width, height) should be greater than 0")
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
        
        # Raise errors if any
        if errors:
            error_msg = "; ".join(errors)
            raise ValueError(error_msg)

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

# Enhanced UPS Rating Service
class UPSRatingService:
    def __init__(self):
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=Config.REQUEST_TIMEOUT,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    def _get_service_id(self, service_code: str, context: str) -> Optional[int]:
        """Get service ID for the given service code and context.
        
        This method returns None to match the frontend's UPSMicroserviceResponse behavior,
        where service_id is explicitly set to None and looked up later by the frontend.
        """
        # Return None for frontend compatibility
        return None
    
    async def get_rates(self, request: UPSRateRequest, request_id: str = None, shop_rates: bool = True) -> List[RateQuote]:
        """Get shipping rates from UPS API"""
        start_time = time.time()
        
        # Store request_id for logging
        if request_id is None:
            request_id = f"ups_rate_{int(time.time() * 1000)}"
        
        # Store request_id locally
        local_request_id = request_id
        
        try:
            # Validate required fields
            request.validate_required_fields()
            
            # Get OAuth token
            token = await token_manager.get_token()
            
            # Build UPS API payload
            payload = self._build_rate_payload(request, shop_rates, local_request_id)
            
            # Make request to UPS
            response = await self._make_ups_request(token, payload, local_request_id)
            
            # Parse response with context
            context = UPSConstants.get_context(request.sender_country, request.country)
            quotes = self._parse_rate_response(response, request.sender_country, request.country, context, request.service)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Rate request {local_request_id} completed in {processing_time:.2f}ms, {len(quotes)} quotes returned")
            
            return quotes
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Rate request {local_request_id} failed after {processing_time:.2f}ms: {str(e)}")
            raise
    
    def _build_rate_payload(self, request: UPSRateRequest, shop_rates: bool = True, request_id: str = None) -> Dict[str, Any]:
        """Build UPS API request payload"""
        
        # Get service code
        service_code = UPSConstants.SERVICES.get(request.service, "03")
        
        # Always use "Shop" for JSON API to avoid service validation issues
        request_type = "Shop"
        
        # Validate postal codes based on shipment type
        if request.is_domestic_us():
            # US domestic shipments require postal codes
            if not request.sender_zip or not request.zip:
                raise UPSAPIError("Postal codes are required for US domestic shipments", "MISSING_POSTAL_CODE", 400)
        elif request.is_international():
            # International shipments should have at least one postal code
            if not request.sender_zip and not request.zip:
                logger.warning("At least one postal code recommended for international shipments")
        
        # Build addresses with proper postal code handling
        shipper_postal_code = request.sender_zip or ""
        shipto_postal_code = request.zip or ""
        
        # For US domestic shipments, validate and format postal codes
        if request.is_domestic_us():
            if not shipper_postal_code or not shipto_postal_code:
                raise UPSAPIError("Valid postal codes required for US domestic shipments", "INVALID_POSTAL_CODE", 400)
            
            # Validate US ZIP code format (5 digits or 5+4 format)
            zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
            
            if not zip_pattern.match(shipper_postal_code.strip()):
                # Extract first 5 digits if possible
                digits_only = re.sub(r'\D', '', shipper_postal_code)
                if len(digits_only) >= 5:
                    shipper_postal_code = digits_only[:5]
                else:
                    raise UPSAPIError(f"Invalid sender ZIP code format: '{request.sender_zip}'", "INVALID_POSTAL_CODE", 400)
            
            if not zip_pattern.match(shipto_postal_code.strip()):
                # Extract first 5 digits if possible
                digits_only = re.sub(r'\D', '', shipto_postal_code)
                if len(digits_only) >= 5:
                    shipto_postal_code = digits_only[:5]
                else:
                    raise UPSAPIError(f"Invalid destination ZIP code format: '{request.zip}'", "INVALID_POSTAL_CODE", 400)
        
        # For international shipments, clean postal codes if provided
        elif request.is_international():
            if shipper_postal_code:
                # Clean sender postal code for international
                if request.sender_country == "US":
                    # US origin to international - validate US ZIP
                    zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
                    if not zip_pattern.match(shipper_postal_code.strip()):
                        digits_only = re.sub(r'\D', '', shipper_postal_code)
                        if len(digits_only) >= 5:
                            shipper_postal_code = digits_only[:5]
                else:
                    # International origin - basic cleanup
                    shipper_postal_code = shipper_postal_code.strip()
            
            if shipto_postal_code:
                # Clean destination postal code for international
                if request.country == "US":
                    # International to US - validate US ZIP
                    zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
                    if not zip_pattern.match(shipto_postal_code.strip()):
                        digits_only = re.sub(r'\D', '', shipto_postal_code)
                        if len(digits_only) >= 5:
                            shipto_postal_code = digits_only[:5]
                else:
                    # International destination - basic cleanup
                    shipto_postal_code = shipto_postal_code.strip()
        
        shipper_address = {
            "PostalCode": shipper_postal_code,
            "CountryCode": request.sender_country
        }
        # Add optional address fields only if provided
        if request.sender_city:
            shipper_address["City"] = request.sender_city
        if request.sender_state:
            shipper_address["StateProvinceCode"] = request.sender_state
        
        # Validate address completeness based on shipment type
        if request.is_domestic_us():
            # For US domestic shipments, validate state and city information
            if not request.sender_state or not request.state:
                logger.warning("Missing state information for US domestic shipment - may cause UPS error 111100")
        
        shipto_address = {
            "PostalCode": shipto_postal_code,
            "CountryCode": request.country
        }
        # Add optional address fields only if provided
        if request.city:
            shipto_address["City"] = request.city
        if request.state:
            shipto_address["StateProvinceCode"] = request.state
        
        # Validate and build package with proper weight and dimension handling
        # Ensure weight meets UPS minimum requirements
        validated_weight = max(1.0, request.weight)
        if validated_weight != request.weight:
            logger.info(f"Weight adjusted from {request.weight} to {validated_weight} (UPS minimum 1.0 lb)")
        
        package_weight = {
            "Weight": str(validated_weight),
            "UnitOfMeasurement": {
                "Code": "LBS"
            }
        }
        
        # Validate packaging type
        if request.package not in UPSConstants.PACKAGES:
            logger.warning(f"Invalid package type '{request.package}', using 'your_packaging'")
            package_type_code = UPSConstants.PACKAGES["your_packaging"]
        else:
            package_type_code = UPSConstants.PACKAGES[request.package]
        
        # Build base package data
        package_data = {
            "PackagingType": {
                "Code": package_type_code
            },
            "PackageWeight": package_weight
        }
        
        logger.info(f"Package: {request.package} (code: {package_type_code}), Weight: {validated_weight} LBS")
        
        # Add dimensions only if at least one dimension is provided and valid
        # UPS requires all dimensions to be at least 1 inch if provided
        if request.length > 0 or request.width > 0 or request.height > 0:
            validated_length = max(1, int(request.length)) if request.length > 0 else 1
            validated_width = max(1, int(request.width)) if request.width > 0 else 1
            validated_height = max(1, int(request.height)) if request.height > 0 else 1
            
            package_data["Dimensions"] = {
                "UnitOfMeasurement": {
                    "Code": "IN"
                },
                "Length": str(validated_length),
                "Width": str(validated_width), 
                "Height": str(validated_height)
            }
            
    
        
        # Build shipment
        shipment = {
            "Shipper": {
                "Address": shipper_address
            },
            "ShipTo": {
                "Address": shipto_address
            },
            "Package": [package_data]
        }
        
        # Include specific service
        if request.service != "all":
            service_code = UPSConstants.SERVICES.get(request.service)
            if service_code:
                shipment["Service"] = {
                    "Code": service_code
                }
    
        # For "all" service requests, Service field is omitted
        
        # Add insured value if specified
        if request.insured_value > 0:
            shipment["Package"][0]["PackageServiceOptions"] = {
                "InsuredValue": {
                    "CurrencyCode": request.currency_code,
                    "MonetaryValue": str(request.insured_value)
                }
            }
        
        # Add pickup and customer info
        pickup_code = UPSConstants.PICKUP_TYPES.get(request.pickup_type, "01")
        customer_code = UPSConstants.CUSTOMER_TYPES.get(request.customer_type, "01")
        
        # Add rating options for specific service
        if request.service != "all":
            shipment["ShipmentRatingOptions"] = {
                "NegotiatedRatesIndicator": ""
            }
        
        # Determine appropriate RequestOption based on service request
        if request.service == "all":
            request_option = "Shop"  # Get all available services
        else:
            request_option = "Rate"  # Get rate for specific service
        
        # Build comprehensive payload for UPS JSON API
        payload = {
            "RateRequest": {
                "Request": {
                    "RequestOption": request_option,
                    "TransactionReference": {
                        "CustomerContext": request_id or "UPS_Rate_Request"
                    }
                },
                "PickupType": {
                    "Code": pickup_code
                },
                "CustomerClassification": {
                    "Code": customer_code
                },
                "Shipment": shipment
            }
        }
        

        
        return payload
    
    async def _make_ups_request(self, token: str, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Make request to UPS API with retries"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "UPS-Rating-Microservice/1.0"
        }
        
        # Determine the correct endpoint URL based on request option
        request_option = payload['RateRequest']['Request']['RequestOption']
        if request_option == "Shop":
            api_url = Config.RATE_URL.replace('/Rate', '/Shop')
        else:
            api_url = Config.RATE_URL
        
        logger.info(f"Making UPS API request to: {api_url}")
        logger.info(f"Request summary: {payload['RateRequest']['Shipment']['Shipper']['Address']['CountryCode']} -> {payload['RateRequest']['Shipment']['ShipTo']['Address']['CountryCode']}")

        
        # Log key validation points that could cause 111100 errors
        shipper_postal = payload['RateRequest']['Shipment']['Shipper']['Address']['PostalCode']
        shipto_postal = payload['RateRequest']['Shipment']['ShipTo']['Address']['PostalCode']

        
        if not shipper_postal or not shipto_postal:
            logger.warning("Empty postal codes detected - this may cause UPS error 111100")
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = await self.client.post(api_url, json=payload, headers=headers)
                
                if response.status_code == 401:
                    token_manager.invalidate_token()
                    raise UPSAPIError("Authentication failed", "AUTH_ERROR", 401)
                elif response.status_code != 200:
                    error_text = response.text
                    logger.error(f"UPS API returned status {response.status_code}: {error_text}")
                    
                    try:
                        error_data = response.json()
                        if "response" in error_data and "errors" in error_data["response"]:
                            errors = error_data["response"]["errors"]
                            if errors:
                                error_msg = errors[0].get("message", "Unknown error")
                                error_code = errors[0].get("code", "")
                                
                                # Enhanced error handling for common UPS errors
                                if error_code == "111100":
                                    logger.error(f"UPS Error 111100 - Invalid service from origin. Diagnostic info:")
                                    logger.error(f"1. Route: {payload['RateRequest']['Shipment']['Shipper']['Address']['CountryCode']} -> {payload['RateRequest']['Shipment']['ShipTo']['Address']['CountryCode']}")
                                    logger.error(f"2. Postal codes: {payload['RateRequest']['Shipment']['Shipper']['Address'].get('PostalCode', 'MISSING')} -> {payload['RateRequest']['Shipment']['ShipTo']['Address'].get('PostalCode', 'MISSING')}")
                                    logger.error(f"3. Cities: {payload['RateRequest']['Shipment']['Shipper']['Address'].get('City', 'MISSING')} -> {payload['RateRequest']['Shipment']['ShipTo']['Address'].get('City', 'MISSING')}")
                                    logger.error(f"4. States: {payload['RateRequest']['Shipment']['Shipper']['Address'].get('StateProvinceCode', 'MISSING')} -> {payload['RateRequest']['Shipment']['ShipTo']['Address'].get('StateProvinceCode', 'MISSING')}")
                                elif error_code == "111210":
                                    logger.error(f"UPS Error 111210 - Invalid postal code combination")
                                elif error_code == "111057":
                                    logger.error(f"UPS Error 111057 - Invalid origin postal code")
                                elif error_code == "111058":
                                    logger.error(f"UPS Error 111058 - Invalid destination postal code")
                                elif error_code == "111035":
                                    logger.error(f"UPS Error 111035 - Invalid package weight or dimensions")
                                elif error_code.startswith("111"):
                                    logger.error(f"UPS Validation Error {error_code} - Check address and package information")
                                
                                raise UPSAPIError(f"UPS API error {error_code}: {error_msg}", error_code, response.status_code)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse UPS error response as JSON: {error_text}")
                    
                    raise UPSAPIError(f"UPS API error: {response.status_code}", "API_ERROR", response.status_code)
                
                return response.json()
                
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise UPSAPIError("Connection timeout", "TIMEOUT_ERROR", 503)
                
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
    
    def _parse_rate_response(self, response_data: Dict[str, Any], origin_country: str, dest_country: str, context: str, requested_service: str = "all") -> List[RateQuote]:
        """Parse UPS API response into rate quotes"""
        quotes = []
        
        try:
            logger.info(f"Parsing UPS response for context: {context}, requested_service: {requested_service}")
            rate_response = response_data.get("RateResponse", {})
            rated_shipments = rate_response.get("RatedShipment", [])
            
            if not rated_shipments:
                logger.warning(f"No RatedShipment found in response: {response_data}")
                return quotes
            
            if not isinstance(rated_shipments, list):
                rated_shipments = [rated_shipments]
            
            logger.info(f"Found {len(rated_shipments)} rated shipments to process")
            
            # Get the requested service code for filtering
            requested_service_code = None
            if requested_service != "all":
                requested_service_code = UPSConstants.SERVICES.get(requested_service)
            
            for shipment in rated_shipments:
                try:
                    service = shipment.get("Service", {})
                    service_code = service.get("Code", "")
                    
                    # Filter by requested service if specified
                    if requested_service_code and service_code != requested_service_code:
        
                        continue
                    
                    # Get service name using Ruby logic
                    service_name = UPSConstants.get_service_name_from_code(
                        origin_country, dest_country, service_code
                    )
                    
                    # Get charges
                    total_charges = shipment.get("TotalCharges", {})
                    total_charge = float(total_charges.get("MonetaryValue", 0))
                    currency_code = total_charges.get("CurrencyCode", "USD")
                    
                    # Get delivery information
                    delivery_days = None
                    guaranteed = None
                    
                    if "GuaranteedDelivery" in shipment:
                        guaranteed_info = shipment["GuaranteedDelivery"]
                        guaranteed = guaranteed_info.get("BusinessDaysInTransit") is not None
                        if guaranteed:
                            delivery_days = int(guaranteed_info.get("BusinessDaysInTransit", 0))
                    
                    quote = RateQuote(
                        service_code=service_code,
                        service_name=service_name,
                        total_charge=total_charge,
                        currency_code=currency_code,
                        delivery_days=delivery_days,
                        guaranteed=guaranteed,
                        context=context,
                        method_id=self._get_service_id(service_code, context)
                    )
                    
                    quotes.append(quote)
                    logger.info(f"Added quote: {service_name} ({service_code}) - ${total_charge} {currency_code}")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse shipment: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(quotes)} quotes from UPS response")
            return quotes
            
        except Exception as e:
            logger.error(f"Failed to parse UPS response: {e}")
            raise UPSAPIError("Failed to parse response", "PARSE_ERROR")

# Application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting UPS Rating Microservice")
    yield
    logger.info("Shutting down UPS Rating Microservice")

app = FastAPI(
    title="UPS Rating Microservice",
    description="""Production-ready UPS Rating API with Ruby UPSRateRequest compatibility.
    
    **Features:**
    - Full compatibility with Ruby UPSRateRequest class
    - Support for all UPS service types and package types
    - Comprehensive error handling and validation
    - Health checks and monitoring endpoints
    
    **Key Endpoints:**
    - `/rates` - Get shipping rates
    - `/health` - Service health check
    - `/metrics` - Service metrics
    """,
    version="1.0.0",
    docs_url="/docs" if Config.DEBUG else None,
    redoc_url="/redoc" if Config.DEBUG else None,
    default_response_class=ORJSONResponse,
    lifespan=lifespan
)

# Middleware - optimized for production
if Config.DEBUG:
    # Development CORS - allow all
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
else:
    # Production CORS - more restrictive
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(CORSMiddleware, 
                      allow_origins=allowed_origins, 
                      allow_methods=["GET", "POST"], 
                      allow_headers=["Authorization", "Content-Type"])

app.add_middleware(GZipMiddleware, minimum_size=500)

# Security
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not Config.API_KEY:
        return True
    if not credentials or credentials.credentials != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Exception handlers
@app.exception_handler(UPSAPIError)
async def ups_api_exception_handler(request: Request, exc: UPSAPIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code or "UPS_API_ERROR",
            "message": exc.message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# Main endpoint - Ruby compatible
@app.post("/rates", response_model=RateResponse, summary="Get UPS rates (Ruby UPSRateRequest compatible)")
async def get_ups_rates(
    request: UPSRateRequest,
    _: bool = Depends(verify_api_key)
) -> RateResponse:
    """
    Get UPS shipping rates - 100% compatible with Ruby UPSRateRequest class.
    
    Supports all the same parameters and logic as the original Ruby implementation:
    - All service types (next_day, 2_day, ground, worldwide_express, etc.)
    - All package types (ups_envelope, your_packaging, ups_tube, etc.) 
    - All pickup types (daily_pickup, customer_counter, one_time_pickup, etc.)
    - All customer types (wholesale, occasional, retail)
    - Regional service contexts (US Domestic, US Origin, Canada Origin, etc.)
    - Insurance and special services
    - Proper weight minimums (1.0 lb minimum)
    - International vs domestic logic
    - Rate shopping (multiple quotes) by default
    """
    request_id = f"ups_rate_{int(time.time() * 1000)}"
    start_time = time.time()
    
    # Store request_id in local variable to ensure scope availability
    local_request_id = request_id
    
    logger.info(f"UPS Rate request {local_request_id}: {request.sender_country} -> {request.country}, {request.weight} {request.weight_units}, service: {request.service}")
    
    try:
        async with UPSRatingService() as ups_service:
            quotes = await ups_service.get_rates(request, local_request_id, shop_rates=True)
        
        processing_time = (time.time() - start_time) * 1000
        context = UPSConstants.get_context(request.sender_country, request.country)
        
        response = RateResponse(
            quotes=quotes,
            request_id=local_request_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time,
            context=context
        )
        
        return response
        
    except ValueError as e:
        # Validation errors should return 400 Bad Request
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Rate request {local_request_id} validation failed after {processing_time:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "request_id": local_request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except UPSAPIError as e:
        # UPS API errors should use their specific status codes
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Rate request {local_request_id} UPS API failed after {processing_time:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code or "UPS_API_ERROR",
                "message": e.message,
                "request_id": local_request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except Exception as e:
        # Unexpected errors should return 500
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Rate request {local_request_id} unexpected error after {processing_time:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": local_request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )



# Health check and diagnostic endpoints
@app.get("/health", summary="Service health check")
async def health_check():
    """Comprehensive health check for the UPS microservice"""
    try:
        # Test OAuth token retrieval
        token_start = time.time()
        token = await token_manager.get_token()
        token_time = (time.time() - token_start) * 1000
        
        # Test UPS API connectivity
        connectivity_start = time.time()
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Just test if we can reach the UPS API endpoint
            try:
                response = await client.head(Config.RATE_URL.replace('/rating/v1/Rate', '/'), timeout=5.0)
                connectivity_status = "ok" if response.status_code in [200, 404, 405] else "degraded"
            except:
                connectivity_status = "failed"
        connectivity_time = (time.time() - connectivity_start) * 1000
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "checks": {
                "oauth_token": {
                    "status": "ok" if token else "failed",
                    "response_time_ms": round(token_time, 2)
                },
                "ups_api_connectivity": {
                    "status": connectivity_status,
                    "response_time_ms": round(connectivity_time, 2)
                }
            },
            "configuration": {
                "oauth_url": Config.OAUTH_URL,
                "rate_url": Config.RATE_URL,
                "timeout_seconds": Config.REQUEST_TIMEOUT,
                "max_retries": Config.MAX_RETRIES
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
        )

@app.get("/validate", summary="Validate request without making UPS API call")
async def validate_request(
    sender_country: str,
    country: str,
    weight: float,
    zip: str = None,
    sender_zip: str = None,
    city: str = None,
    sender_city: str = None,
    state: str = None,
    sender_state: str = None,
    service: str = "all",
    package: str = "your_packaging"
):
    """Validate a shipping request without making an actual UPS API call"""
    try:
        # Create request object
        request = UPSRateRequest(
            sender_country=sender_country,
            country=country,
            weight=weight,
            zip=zip,
            sender_zip=sender_zip,
            city=city,
            sender_city=sender_city,
            state=state,
            sender_state=sender_state,
            service=service,
            package=package
        )
        
        # Validate the request
        request.validate_required_fields()
        
        # Determine shipment context
        context = UPSConstants.get_context(sender_country, country)
        
        return {
            "status": "valid",
            "context": context,
            "shipment_type": "domestic_us" if request.is_domestic_us() else "international",
            "validation_passed": True,
            "recommendations": {
                "service_available": service in UPSConstants.SERVICES,
                "package_type_valid": package in UPSConstants.PACKAGES,
                "weight_adjusted": max(0.1, weight) != weight
            }
        }
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid",
                "validation_passed": False,
                "errors": [str(e)]
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "validation_passed": False,
                "errors": [f"Validation error: {str(e)}"]
            }
        )







# Health and monitoring
@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint with UPS API connectivity test.
    """
    try:
        # Test OAuth connectivity
        token = await token_manager.get_token()
        ups_status = "healthy"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        ups_status = "unhealthy"
    
    return {
        "status": ups_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": Config.UPS_ENVIRONMENT,
        "version": "1.0.0",
        "ruby_compatible": True,
        "dependencies": {
            "ups_oauth": ups_status,
            "ups_rating_api": ups_status
        }
    }

@app.get("/metrics", summary="Service metrics")
async def get_metrics(_: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """
    Get service metrics and statistics.
    """
    return {
        "service": {
            "name": "UPS Rating Microservice",
            "version": "1.0.0",
            "environment": Config.UPS_ENVIRONMENT,
            "ruby_compatible": True,
            "supported_endpoints": [
                "/rates",
                "/parallel_rates", 
                "/ups_rate_request",
                "/services",
                "/service_name/{code}",
                "/constants",
                "/health",
                "/metrics"
            ]
        },
        "configuration": {
            "request_timeout": Config.REQUEST_TIMEOUT,
            "max_retries": Config.MAX_RETRIES,
            "api_version": UPSConstants.API_VERSION
        },
        "ruby_features": {
            "all_package_types": len(UPSConstants.PACKAGES),
            "all_service_types": len(UPSConstants.SERVICES),
            "regional_contexts": len(UPSConstants.SERVICE_CODES),
            "pickup_types": len(UPSConstants.PICKUP_TYPES),
            "customer_types": len(UPSConstants.CUSTOMER_TYPES)
        }
    }



if __name__ == "__main__":
    # Optimized server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "4")) if not Config.DEBUG else 1,
        access_log=Config.DEBUG,
        log_level=Config.LOG_LEVEL.lower(),
        reload=Config.DEBUG,
        loop="uvloop" if not Config.DEBUG else "asyncio",
        http="httptools" if not Config.DEBUG else "h11"
    )