class USPSAPIError(Exception):
  def __init__(self, message: str, error_code: str = None, status_code: int = 500):
    self.message = message
    self.error_code = error_code
    self.status_code = status_code
    super().__init__(self.message)
