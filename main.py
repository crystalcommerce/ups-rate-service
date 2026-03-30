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

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
from routers import all_routers
from core.security import require_api_key
from core.logging import get_logger
from core.constants.ups import UPSConstants
from core.config import settings
from core.exceptions import UPSAPIError, USPSAPIError

logger = logging.getLogger(__name__)

# Set logging level with validation
try:
    log_level = settings.LOG_LEVEL.strip() if settings.LOG_LEVEL else "INFO"
    if hasattr(logging, log_level):
        logging.getLogger().setLevel(getattr(logging, log_level))
        logger.info(f"Logging level set to: {log_level}")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.warning(f"Invalid log level '{log_level}', defaulting to INFO")
except Exception as e:
    logging.getLogger().setLevel(logging.INFO)
    logger.error(f"Error setting log level: {e}, defaulting to INFO")


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
        credentials = f"{settings.UPS_CLIENT_ID}:{settings.UPS_CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {encoded_credentials}"
        }
        data = {"grant_type": "client_credentials"}
        
        logger.info(f"Requesting OAuth token from: {settings.UPS_OAUTH_URL}")
        
        async with httpx.AsyncClient(timeout=settings.UPS_REQUEST_TIMEOUT) as client:
            response = await client.post(settings.UPS_OAUTH_URL, data=data, headers=headers)
            
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

class USPSTokenManager:
    def __init__(self):
        self._cache = {}
        self._lock = asyncio.Lock()

    async def get_token(self, scope: str = "addresses") -> str:
        """Get OAuth token for the specified scope.

        Tokens expire in 8 hours (28800 seconds).

        Valid scopes:
        - "addresses" - for address validation
        - "domestic-prices" - for domestic rate calculation
        - "international-prices" - for international rate calculation
        """
        async with self._lock:
            current_time = time.time()
            if scope in self._cache:
                cached = self._cache[scope]
                if cached["access_token"] and cached["expires_at"] > current_time:
                    return cached["access_token"]

            token_data = await self._request_token(scope)
            expires_in = int(token_data.get("expires_in", 28800))

            self._cache[scope] = {
                "access_token": token_data["access_token"],
                "expires_at": current_time + expires_in - 60
            }
            return self._cache[scope]["access_token"]

    async def _request_token(self, scope: str) -> Dict[str, Any]:
        """Request OAuth token using Client Credentials flow"""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials",
            "client_id": settings.USPS_CLIENT_ID,
            "client_secret": settings.USPS_CLIENT_SECRET,
            "scope": scope
        }
        logger.info(f"Requesting USPS OAuth token from: {settings.USPS_OAUTH_URL} with scope: {scope}")
        async with httpx.AsyncClient(timeout=settings.USPS_REQUEST_TIMEOUT) as client:
            response = await client.post(settings.USPS_OAUTH_URL, data=data, headers=headers)
            if response.status_code != 200:
                logger.error(f"USPS OAuth failed with status {response.status_code}: {response.text}")
                raise USPSAPIError(f"USPS OAuth failed: {response.status_code}", "AUTH_ERROR", response.status_code)
            token_data = response.json()
            if "access_token" not in token_data:
                logger.error(f"Invalid USPS OAuth response: {token_data}")
                raise USPSAPIError("Invalid USPS OAuth response", "INVALID_RESPONSE")
            logger.info("Successfully obtained USPS OAuth token")
            return token_data

    def invalidate_token(self, scope: str = None):
        """Invalidate token(s). If scope is None, invalidates all tokens."""
        if scope:
            if scope in self._cache:
                del self._cache[scope]
        else:
            self._cache.clear()

usps_token_manager = USPSTokenManager()

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
            timeout=settings.UPS_REQUEST_TIMEOUT,
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
            api_url = settings.UPS_RATE_URL.replace('/Rate', '/Shop')
        else:
            api_url = settings.UPS_RATE_URL
        
        logger.info(f"Making UPS API request to: {api_url}")
        logger.info(f"Request summary: {payload['RateRequest']['Shipment']['Shipper']['Address']['CountryCode']} -> {payload['RateRequest']['Shipment']['ShipTo']['Address']['CountryCode']}")

        
        # Log key validation points that could cause 111100 errors
        shipper_postal = payload['RateRequest']['Shipment']['Shipper']['Address']['PostalCode']
        shipto_postal = payload['RateRequest']['Shipment']['ShipTo']['Address']['PostalCode']

        
        if not shipper_postal or not shipto_postal:
            logger.warning("Empty postal codes detected - this may cause UPS error 111100")
        
        for attempt in range(settings.UPS_MAX_RETRIES):
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
                if attempt == settings.UPS_MAX_RETRIES - 1:
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

class USPSAddressService:
    def __init__(self):
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=settings.USPS_REQUEST_TIMEOUT,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    async def validate_address(self, street_address: str, city: str, state: str) -> Dict[str, str]:
        """Validate address and return ZIP code information"""
        token = await usps_token_manager.get_token(scope="addresses")
        params = {
            "streetAddress": street_address,
            "city": city,
            "state": state
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        logger.info(f"Validating USPS address: {street_address}, {city}, {state}")
        for attempt in range(settings.USPS_MAX_RETRIES):
            try:
                response = await self.client.get(
                    settings.USPS_ADDRESS_VALIDATION_URL,
                    params=params,
                    headers=headers
                )
                if response.status_code == 401:
                    error_text = response.text
                    logger.error(f"USPS Address API authentication failed (401). Response: {error_text}")
                    usps_token_manager.invalidate_token(scope="addresses")
                    raise USPSAPIError("Authentication failed", "AUTH_ERROR", 401)
                elif response.status_code != 200:
                    error_text = response.text
                    logger.error(f"USPS Address API returned status {response.status_code}: {error_text}")
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_info = error_data["error"]
                            error_msg = error_info.get("message", "Unknown error")
                            error_code = error_info.get("code", "")
                            raise USPSAPIError(f"USPS API error {error_code}: {error_msg}", error_code, response.status_code)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse USPS error response as JSON: {error_text}")
                    raise USPSAPIError(f"USPS API error: {response.status_code}", "API_ERROR", response.status_code)
                response_data = response.json()

                address = response_data.get("address", {})
                zip_code = address.get("ZIPCode", "")
                zip_plus4 = address.get("ZIPPlus4")

                result = {
                    "zipCode": zip_code
                }
                if zip_plus4:
                    result["zipPlus4"] = zip_plus4

                logger.info(f"USPS address validation successful: ZIP={zip_code}, ZIP+4={zip_plus4}")
                return result
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt == settings.USPS_MAX_RETRIES - 1:
                    raise USPSAPIError("Connection timeout", "TIMEOUT_ERROR", 503)
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

class USPSRateService:
    def __init__(self):
        self.client = None
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=settings.USPS_REQUEST_TIMEOUT,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def _is_domestic(self, sender_country: str, destination_country: str) -> bool:
        """Check if shipment is domestic (both origin and destination must be US)"""
        return sender_country.upper() == "US" and destination_country.upper() == "US"

    def _build_letter_payload(self, request: UPSRateRequest) -> Optional[Dict[str, Any]]:
        """
        Build USPS LetterRatesQuery payload for First-Class Mail letters.

        We always attempt to call the letter-rates API for domestic shipments. If dimensions
        are missing or zero we use default letter dimensions so we don't silently skip
        First-Class; USPS will return rates when eligible or we simply get parcel-only.
        """
        if not self._is_domestic(request.sender_country, request.country):
            return None

        logger.info(f"USPS LetterRatesQuery request: {request}")
        weight_oz = max(0.0, float(request.weight)) * 16.0
        logger.info(f"USPS LetterRatesQuery weight: {weight_oz} oz")
        if weight_oz <= 0 or weight_oz > 3.5:
            return None


        length = float(request.length or 0)
        height = float(request.height or 0)
        width = float(request.width or 0)

        if length > 0 and height > 0 and width > 0:
            dims = sorted([length, height, width])
            thickness, height, length = dims[0], dims[1], dims[2]
        else:
            length = 6.0
            height = 4.0
            thickness = 0.02

        payload = {
            "weight": weight_oz,
            "length": length,
            "height": height,
            "thickness": thickness,
            "processingCategory": "LETTERS",
        }
        logger.info(f"USPS LetterRatesQuery payload: {payload}")

        return payload

    def _build_rate_payload(self, request: UPSRateRequest) -> Dict[str, Any]:
        """Build USPS API request payload from UPSRateRequest format"""
        is_domestic = self._is_domestic(request.sender_country, request.country)

        if not request.sender_zip:
            raise ValueError("sender_zip is required for USPS rate calculation")
        if not request.zip and is_domestic:
            raise ValueError("zip is required for USPS domestic rate calculation")

        payload = {
            "originZIPCode": request.sender_zip,
            "weight": request.weight,
            "length": max(1, float(request.length)) if request.length > 0 else 1.0,
            "width": max(1, float(request.width)) if request.width > 0 else 1.0,
            "height": max(1, float(request.height)) if request.height > 0 else 1.0,
            "priceType": "COMMERCIAL"
        }

        if is_domestic:
            payload["destinationZIPCode"] = request.zip
        else:
            payload["destinationCountryCode"] = request.country
            if request.zip:
                payload["foreignPostalCode"] = request.zip

        return payload

    async def get_rates(self, request: UPSRateRequest, request_id: str = None) -> List[RateQuote]:
        """Get shipping rates from USPS API"""
        start_time = time.time()

        if request_id is None:
            request_id = f"usps_rate_{int(time.time() * 1000)}"

        local_request_id = request_id

        try:
            is_domestic = self._is_domestic(request.sender_country, request.country)

            scope = "domestic-prices" if is_domestic else "international-prices"
            token = await usps_token_manager.get_token(scope=scope)

            payload = self._build_rate_payload(request)

            if is_domestic:
                api_url = settings.USPS_DOMESTIC_RATES_URL
            else:
                api_url = settings.USPS_INTERNATIONAL_RATES_URL

            response = await self._make_usps_request(token, payload, api_url, local_request_id)

            quotes = self._parse_rate_response(response, is_domestic)

            if is_domestic:
                letter_payload = self._build_letter_payload(request)
                if letter_payload:
                    logger.info(f"[LETTER-RATES] [{local_request_id}] Eligible for First-Class Mail - calling letter-rates API")
                    try:
                        letter_response = await self._make_usps_letter_request(token, letter_payload, settings.USPS_LETTER_RATES_URL, local_request_id)
                        letter_quotes = self._parse_letter_rate_response(letter_response)
                        logger.info(f"[LETTER-RATES] [{local_request_id}] Added {len(letter_quotes)} First-Class Mail quote(s) to results")
                        quotes.extend(letter_quotes)
                    except USPSAPIError as e:
                        logger.warning(f"[LETTER-RATES] [{local_request_id}] First-Class Mail request failed: {e}")
                    except Exception as e:
                        logger.warning(f"[LETTER-RATES] [{local_request_id}] Unexpected error fetching letter rates: {e}")
                else:
                    weight_oz = max(0.0, float(request.weight)) * 16.0
                    logger.info(f"[LETTER-RATES] [{local_request_id}] Skipped - weight={weight_oz:.2f}oz (must be >0 and <=3.5oz), sender={request.sender_country}, dest={request.country}")

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"USPS Rate request {local_request_id} completed in {processing_time:.2f}ms, {len(quotes)} quotes returned")

            return quotes

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"USPS Rate request {local_request_id} failed after {processing_time:.2f}ms: {str(e)}")
            raise

    async def _make_usps_request(self, token: str, payload: Dict[str, Any], api_url: str, request_id: str) -> Dict[str, Any]:
        """Make request to USPS API with retries"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "USPS-Rating-Microservice/1.0"
        }

        logger.info(f"Making USPS API request to: {api_url}")
        logger.info(f"Request summary: {payload.get('originZIPCode', 'N/A')} -> {payload.get('destinationZIPCode', 'N/A')} ({payload.get('destinationCountryCode', 'US')})")

        for attempt in range(settings.USPS_MAX_RETRIES):
            try:
                response = await self.client.post(api_url, json=payload, headers=headers)

                if response.status_code == 401:
                    error_text = response.text
                    logger.error(f"USPS API authentication failed (401). Response: {error_text}")

                    scope = "domestic-prices" if "prices" in api_url and "international" not in api_url else "international-prices"
                    usps_token_manager.invalidate_token(scope=scope)

                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_info = error_data["error"]
                            error_msg = error_info.get("message", "Unknown error")
                            error_code = error_info.get("code", "")
                            raise USPSAPIError(
                                f"USPS API authentication error {error_code}: {error_msg}",
                                error_code,
                                response.status_code,
                            )
                    except json.JSONDecodeError:
                        logger.error(
                            "Failed to parse USPS authentication error response as JSON: "
                            f"{error_text}"
                        )

                    raise USPSAPIError("USPS API authentication failed - token may be invalid or lack required permissions", "AUTH_ERROR", 401)
                elif response.status_code != 200:
                    error_text = response.text
                    logger.error(f"USPS API returned status {response.status_code}: {error_text}")

                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_info = error_data["error"]
                            error_msg = error_info.get("message", "Unknown error")
                            error_code = error_info.get("code", "")
                            raise USPSAPIError(f"USPS API error {error_code}: {error_msg}", error_code, response.status_code)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse USPS error response as JSON: {error_text}")

                    raise USPSAPIError(f"USPS API error: {response.status_code}", "API_ERROR", response.status_code)

                return response.json()

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt == settings.USPS_MAX_RETRIES - 1:
                    raise USPSAPIError("Connection timeout", "TIMEOUT_ERROR", 503)

                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
    async def _make_usps_letter_request(self, token: str, payload: Dict[str, Any], api_url: str, request_id: str) -> Dict[str, Any]:
        """Make request to USPS Letter Rates API with retries"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "USPS-Rating-Microservice/1.0"
        }

        logger.info(f"[LETTER-RATES] [{request_id}] Sending request to: {api_url}")
        logger.info(f"[LETTER-RATES] [{request_id}] Payload: {json.dumps(payload)}")

        for attempt in range(settings.USPS_MAX_RETRIES):
            try:
                logger.info(f"[LETTER-RATES] [{request_id}] Attempt {attempt + 1}/{settings.USPS_MAX_RETRIES}")
                response = await self.client.post(api_url, json=payload, headers=headers)

                logger.info(f"[LETTER-RATES] [{request_id}] Response status: {response.status_code}")
                logger.info(f"[LETTER-RATES] [{request_id}] Raw response body: {response.text}")

                if response.status_code == 401:
                    error_text = response.text
                    logger.error(f"[LETTER-RATES] [{request_id}] Authentication failed (401): {error_text}")
                    usps_token_manager.invalidate_token(scope="domestic-prices")

                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_info = error_data["error"]
                            error_msg = error_info.get("message", "Unknown error")
                            error_code = error_info.get("code", "")
                            raise USPSAPIError(
                                f"USPS Letter Rates API authentication error {error_code}: {error_msg}",
                                error_code,
                                response.status_code,
                            )
                    except json.JSONDecodeError:
                        logger.error(
                            f"[LETTER-RATES] [{request_id}] Failed to parse auth error response as JSON: "
                            f"{error_text}"
                        )

                    raise USPSAPIError("USPS Letter Rates API authentication failed", "AUTH_ERROR", 401)
                elif response.status_code != 200:
                    error_text = response.text
                    logger.error(f"[LETTER-RATES] [{request_id}] API returned status {response.status_code}: {error_text}")

                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_info = error_data["error"]
                            error_msg = error_info.get("message", "Unknown error")
                            error_code = error_info.get("code", "")
                            raise USPSAPIError(f"USPS Letter Rates API error {error_code}: {error_msg}", error_code, response.status_code)
                    except json.JSONDecodeError:
                        logger.error(f"[LETTER-RATES] [{request_id}] Failed to parse error response as JSON: {error_text}")

                    raise USPSAPIError(f"USPS Letter Rates API error: {response.status_code}", "API_ERROR", response.status_code)

                response_data = response.json()
                logger.info(f"[LETTER-RATES] [{request_id}] Success - parsed response: {json.dumps(response_data)}")
                return response_data

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.error(f"[LETTER-RATES] [{request_id}] Connection error on attempt {attempt + 1}: {e}")
                if attempt == settings.USPS_MAX_RETRIES - 1:
                    raise USPSAPIError("Connection timeout", "TIMEOUT_ERROR", 503)

                wait_time = 2 ** attempt
                logger.info(f"[LETTER-RATES] [{request_id}] Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    def _parse_rate_response(self, response_data: Dict[str, Any], is_domestic: bool) -> List[RateQuote]:
        """Parse USPS API response into rate quotes matching UPS format"""
        quotes = []

        try:
            logger.info(f"Parsing USPS response for {'domestic' if is_domestic else 'international'}")

            rate_options = response_data.get("rateOptions", [])

            if not rate_options:
                logger.warning(f"No rateOptions found in USPS response: {response_data}")
                return quotes

            if not isinstance(rate_options, list):
                rate_options = [rate_options]

            logger.info(f"Found {len(rate_options)} rate options to process")

            for rate_option in rate_options:
                try:
                    rates = rate_option.get("rates", [])
                    if not isinstance(rates, list):
                        rates = [rates]

                    for rate in rates:
                        mail_class = rate.get("mailClass", "")
                        rate_indicator = rate.get("rateIndicator", "")
                        product_name = rate.get("productName", mail_class)
                        total_base_price = rate_option.get("totalBasePrice", 0)

                        service_code = f"{mail_class}|{rate_indicator}" if rate_indicator else mail_class

                        context = "Domestic" if is_domestic else "International"

                        quote = RateQuote(
                            service_code=service_code,
                            service_name=product_name,
                            total_charge=float(total_base_price),
                            currency_code="USD",
                            context=context
                        )

                        quotes.append(quote)
                        logger.info(f"Added USPS quote: {product_name} ({service_code}) - ${total_base_price}")

                except Exception as e:
                    logger.warning(f"Failed to parse USPS rate option: {e}")
                    continue

            logger.info(f"Successfully parsed {len(quotes)} quotes from USPS response")
            return quotes

        except Exception as e:
            logger.error(f"Failed to parse USPS response: {e}")
            raise USPSAPIError("Failed to parse response", "PARSE_ERROR")

    def _parse_letter_rate_response(self, response_data: Dict[str, Any]) -> List[RateQuote]:
        """Parse USPS Letter Rates API response into RateQuote list for First-Class Mail."""
        quotes: List[RateQuote] = []

        try:
            logger.info(f"[LETTER-RATES] Parsing letter rate response")
            total_base_price = response_data.get("totalBasePrice", 0)
            rates = response_data.get("rates", [])
            logger.info(f"[LETTER-RATES] totalBasePrice={total_base_price}, rates count={len(rates) if isinstance(rates, list) else 1}")

            if not isinstance(rates, list):
                rates = [rates]

            for rate in rates:
                try:
                    product_name = rate.get("description", "First-Class Mail")
                    mail_class = rate.get("mailClass", "FIRST-CLASS_MAIL")
                    price = float(total_base_price or rate.get("price", 0))

                    service_code = mail_class

                    logger.info(f"[LETTER-RATES] Parsed rate: mailClass={mail_class}, description={product_name}, price=${price}")

                    quote = RateQuote(
                        service_code=service_code,
                        service_name=product_name,
                        total_charge=price,
                        currency_code="USD",
                        context="Domestic",
                    )

                    quotes.append(quote)

                except Exception as e:
                    logger.warning(f"[LETTER-RATES] Failed to parse letter rate entry: {e}, raw entry: {rate}")
                    continue

            logger.info(f"[LETTER-RATES] Finished parsing - {len(quotes)} First-Class Mail quote(s) produced")
            return quotes

        except Exception as e:
            logger.error(f"[LETTER-RATES] Failed to parse letter rates response: {e}")
            raise USPSAPIError("Failed to parse letter rates response", "PARSE_ERROR")

# Application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting UPS Rating Microservice")
    yield
    logger.info("Shutting down UPS Rating Microservice")

app = FastAPI(
    title="UPS Rating Microservice",
    description="UPS Rating API",
    version="1.0.0",
    lifespan=lifespan
)

for router in all_routers:
  app.include_router(
    router,
    dependencies=[Depends(require_api_key)]
  )

# CORS middleware
app.add_middleware(CORSMiddleware, 
                  allow_origins=["*"], 
                  allow_methods=["GET", "POST"], 
                  allow_headers=["*"])

# Security
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not settings.API_KEY:
        return True
    if not credentials or credentials.credentials != settings.API_KEY:
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

@app.exception_handler(USPSAPIError)
async def usps_api_exception_handler(request: Request, exc: USPSAPIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code or "USPS_API_ERROR",
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

# USPS Address Validation Endpoint
@app.get("/usps/addresses/validate", summary="Validate USPS address and get ZIP code")
async def validate_usps_address(
    streetAddress: str,
    city: str,
    state: str,
    _: bool = Depends(verify_api_key)
) -> Dict[str, str]:
    """
    Validate a USPS address and return ZIP code information.
    Query parameters:
    - streetAddress: Street address
    - city: City name
    - state: Two-character state code

    Returns:
    - zipCode: 5-digit ZIP code
    - zipPlus4: 4-digit ZIP+4 code (if available)
    """
    request_id = f"usps_address_{int(time.time() * 1000)}"
    start_time = time.time()

    logger.info(f"USPS Address validation request {request_id}: {streetAddress}, {city}, {state}")

    try:
        async with USPSAddressService() as address_service:
            result = await address_service.validate_address(streetAddress, city, state)

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"USPS Address validation {request_id} completed in {processing_time:.2f}ms")

        return result

    except USPSAPIError as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"USPS Address validation {request_id} failed after {processing_time:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code or "USPS_API_ERROR",
                "message": e.message,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"USPS Address validation {request_id} unexpected error after {processing_time:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@app.post("/usps/rates", response_model=RateResponse, summary="Get USPS rates (matching UPS format)")
async def get_usps_rates(
    request: UPSRateRequest,
    _: bool = Depends(verify_api_key)
) -> RateResponse:
    """
    Get USPS shipping rates - compatible with UPS microservice request format.

    Supports both domestic and international rates:
    - Domestic: Uses USPS domestic prices API
    - International: Uses USPS international prices API

    Returns rates in the same format as UPS microservice:
    {
        "quotes": [
            {
                "service_code": "PRIORITY_MAIL|SP",
                "service_name": "Priority Mail",
                "total_charge": 12.50,
                "context": "Domestic" or "International"
            }
        ]
    }
    """
    request_id = f"usps_rate_{int(time.time() * 1000)}"
    start_time = time.time()

    logger.info(f"USPS Rate request {request_id}: {request.sender_country} -> {request.country}, {request.weight} lbs")

    try:
        request.validate_required_fields()

        async with USPSRateService() as usps_service:
            quotes = await usps_service.get_rates(request, request_id)

        processing_time = (time.time() - start_time) * 1000
        is_domestic = request.country.upper() == "US" and request.sender_country.upper() == "US"
        context_str = "Domestic" if is_domestic else "International"

        response = RateResponse(
            quotes=quotes,
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time,
            context=context_str
        )

        return response

    except ValueError as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"USPS Rate request {request_id} validation failed after {processing_time:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except USPSAPIError as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"USPS Rate request {request_id} USPS API failed after {processing_time:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code or "USPS_API_ERROR",
                "message": e.message,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"USPS Rate request {request_id} unexpected error after {processing_time:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": request_id,
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
                response = await client.head(settings.RATE_URL.replace('/rating/v1/Rate', '/'), timeout=5.0)
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
                "oauth_url": settings.UPS_OAUTH_URL,
                "rate_url": settings.UPS_RATE_URL,
                "timeout_seconds": settings.UPS_REQUEST_TIMEOUT,
                "max_retries": settings.UPS_MAX_RETRIES
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
# Duplicate health endpoint removed - using the comprehensive one above

@app.get("/metrics", summary="Service metrics")
async def get_metrics(_: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """
    Get service metrics and statistics.
    """
    return {
        "service": {
            "name": "UPS Rating Microservice",
            "version": "1.0.0",
            "environment": settings.UPS_ENVIRONMENT,
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
            "request_timeout": settings.UPS_REQUEST_TIMEOUT,
            "max_retries": settings.UPS_MAX_RETRIES,
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
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )