import os


def require_env(key: str) -> str:
  value = os.getenv(key)
  if not value:
    raise RuntimeError(f"Missing required environment variable: {key}")
  return value


class Settings:
  # -------------------------
  # General
  # -------------------------
  DEBUG = os.getenv("DEBUG", "false").lower() == "true"
  LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
  AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")


  # -------------------------
  # UPS (REQUIRED)
  # -------------------------
  UPS_CLIENT_ID = require_env("UPS_CLIENT_ID")
  UPS_CLIENT_SECRET = require_env("UPS_CLIENT_SECRET")
  UPS_ENVIRONMENT = os.getenv("UPS_ENVIRONMENT", "sandbox").lower()
  UPS_REQUEST_TIMEOUT = int(os.getenv("UPS_REQUEST_TIMEOUT", "30"))
  UPS_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

  UPS_RATE_LIMIT = int(os.getenv("UPS_RATE_LIMIT", "100"))

  if UPS_ENVIRONMENT == "production":
    UPS_OAUTH_URL = "https://onlinetools.ups.com/security/v1/oauth/token"
    UPS_RATE_URL = "https://onlinetools.ups.com/api/rating/v1/Rate"
  else:
    UPS_OAUTH_URL = "https://wwwcie.ups.com/security/v1/oauth/token"
    UPS_RATE_URL = "https://wwwcie.ups.com/api/rating/v1/Rate"

  # -------------------------
  # FedEx (REQUIRED)
  # -------------------------
  FEDEX_CLIENT_ID = require_env("FEDEX_CLIENT_ID")
  FEDEX_CLIENT_SECRET = require_env("FEDEX_CLIENT_SECRET")
  FEDEX_ACCOUNT_NUMBER = require_env("FEDEX_ACCOUNT_NUMBER")

  FEDEX_BASE_URL = os.getenv("FEDEX_BASE_URL", "https://apis-sandbox.fedex.com")

  # -------------------------
  # USPS (OPTIONAL)
  # -------------------------
  USPS_CLIENT_ID = os.getenv("USPS_CLIENT_ID", "")
  USPS_CLIENT_SECRET = os.getenv("USPS_CLIENT_SECRET", "")

  USPS_REQUEST_TIMEOUT = int(os.getenv("USPS_REQUEST_TIMEOUT", "30"))
  USPS_MAX_RETRIES = int(os.getenv("USPS_MAX_RETRIES", "3"))

  USPS_OAUTH_URL = os.getenv(
    "USPS_OAUTH_URL",
    "https://apis.usps.com/oauth2/v3/token"
  )
  USPS_ADDRESS_VALIDATION_URL = os.getenv(
    "USPS_ADDRESS_VALIDATION_URL",
    "https://apis.usps.com/addresses/v3/zipcode"
  )
  USPS_DOMESTIC_RATES_URL = os.getenv(
    "USPS_DOMESTIC_RATES_URL",
    "https://apis.usps.com/prices/v3/total-rates/search"
  )
  USPS_LETTER_RATES_URL = os.getenv(
    "USPS_LETTER_RATES_URL",
    "https://apis.usps.com/prices/v3/letter-rates/search"
  )
  USPS_INTERNATIONAL_RATES_URL = os.getenv(
    "USPS_INTERNATIONAL_RATES_URL",
    "https://apis.usps.com/international-prices/v3/total-rates/search"
  )


settings = Settings()
