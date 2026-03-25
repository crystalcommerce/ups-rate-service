from fastapi import APIRouter, HTTPException
from services.fedex_service import FedexService

router = APIRouter(prefix="/fedex", tags=["FedEx"])

service = FedexService()


@router.post("/rates")
def get_fedex_rates(payload: dict):
  try:
    return service.get_rates(payload)
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
