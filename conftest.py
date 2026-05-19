"""Pytest bootstrap.

`core.config` calls `require_env` at import time, so any test that imports a
service module needs credentials present in the environment. Inject harmless
dummy values here. The presence of this root-level conftest also puts the
project root on `sys.path`, so tests can `import schemas...` / `import services...`.
"""
import os

for _key in (
  "UPS_CLIENT_ID", "UPS_CLIENT_SECRET",
  "FEDEX_CLIENT_ID", "FEDEX_CLIENT_SECRET", "FEDEX_ACCOUNT_NUMBER",
  "USPS_CLIENT_ID", "USPS_CLIENT_SECRET",
):
  os.environ.setdefault(_key, "test-credential")
