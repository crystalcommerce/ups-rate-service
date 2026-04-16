from .ups_router import router as ups_router
from .usps_router import router as usps_router
from .health_router import router as health_router
from .metrics_router import router as metrics_router

secured_routers = [
  ups_router,
  usps_router,
  metrics_router
]

public_routers = [
  health_router
]
