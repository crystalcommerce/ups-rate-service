import logging
import sys
from core.config import settings

def setup_logging():
  level_name = settings.LOG_LEVEL.strip() if settings.LOG_LEVEL else "INFO"
  log_level = getattr(logging, level_name.upper(), logging.INFO)

  logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
  )

  logger = logging.getLogger("core")
  logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")

def get_logger(name: str):
  return logging.getLogger(name)
