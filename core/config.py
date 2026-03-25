import os

FEDEX_CLIENT_ID = os.environ["FEDEX_CLIENT_ID"]
FEDEX_CLIENT_SECRET = os.environ["FEDEX_CLIENT_SECRET"]
FEDEX_BASE_URL = os.environ.get("FEDEX_BASE_URL", "https://apis-sandbox.fedex.com") # Fallback to sandbox mode
FEDEX_ACCOUNT_NUMBER = os.environ["FEDEX_ACCOUNT_NUMBER"]
