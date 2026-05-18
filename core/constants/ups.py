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
