"""Every synchronous carrier call has a timeout, and nothing blocking runs on the event loop.

2026-11-23: the service froze for 16 minutes at a time (box2, four times in a day). A token or
FedEx request made with `requests` and no timeout waited for the kernel to drop a silent
connection (tcp_retries2, ~15.5 min) while blocking uvicorn's only event loop, so every
quote and every health check waited with it.
"""
import asyncio
import importlib
import inspect
import threading
import time

import pytest

from core.config import settings
import auth.ups_oauth as ups_oauth_mod
import auth.usps_oauth as usps_oauth_mod
import auth.fedex_oauth as fedex_oauth_mod
import services.fedex_service as fedex_service_mod
import services.ups_service as ups_service_mod
# routers/__init__ exports each APIRouter under its module's name, so load the modules themselves.
fedex_router_mod = importlib.import_module("routers.fedex_router")
health_router_mod = importlib.import_module("routers.health_router")


class _Resp:
  status_code = 200

  def json(self):
    return {"access_token": "t", "expires_in": 3600}

  def raise_for_status(self):
    pass


@pytest.fixture
def captured(monkeypatch):
  calls = []

  def fake_post(*args, **kwargs):
    calls.append(kwargs)
    return _Resp()

  for mod in (ups_oauth_mod, usps_oauth_mod, fedex_oauth_mod, fedex_service_mod):
    monkeypatch.setattr(mod.requests, "post", fake_post)
  return calls


def test_outbound_timeout_is_bounded():
  connect, read = settings.OUTBOUND_TIMEOUT
  assert 0 < connect <= 5 and 0 < read <= 15


def test_token_fetches_pass_a_timeout(captured):
  ups_oauth_mod.UPSOauth()._generate_token()
  usps_oauth_mod.USPSOauth()._generate_token("prices")
  fedex_oauth_mod.FedexOauth()._generate_token()
  assert len(captured) == 3
  assert all(c.get("timeout") == settings.OUTBOUND_TIMEOUT for c in captured)


def test_fedex_rate_call_passes_a_timeout(captured, monkeypatch):
  svc = fedex_service_mod.FedexService()
  monkeypatch.setattr(svc.auth, "get_token", lambda: "t")
  svc._make_request_with_retry("https://example.invalid/rates", {}, "r1")
  assert captured[-1].get("timeout") == settings.OUTBOUND_TIMEOUT


def test_fedex_route_runs_in_the_threadpool():
  # A plain def route runs in FastAPI's threadpool; an async def would run its blocking
  # requests/time.sleep on the event loop.
  assert not inspect.iscoroutinefunction(fedex_router_mod.get_fedex_rates)


async def test_slow_ups_token_does_not_block_the_event_loop(monkeypatch):
  svc = ups_service_mod.UPSRatingService()
  token_threads = []

  def slow_token():
    token_threads.append(threading.current_thread())
    time.sleep(0.5)
    return "t"

  monkeypatch.setattr(svc.auth, "get_token", slow_token)

  async def fake_request(token, payload, request_id):
    raise RuntimeError("stop after token")

  monkeypatch.setattr(svc, "_make_ups_request", fake_request)
  ticks = 0

  async def ticker():
    nonlocal ticks
    for _ in range(10):
      await asyncio.sleep(0.02)
      ticks += 1

  from schemas.rate_request import RateRequest
  request = RateRequest(sender_country="US", country="US", sender_zip="10001", zip="90210", weight=1)
  results = await asyncio.gather(svc.get_rates(request), ticker(), return_exceptions=True)
  assert token_threads and token_threads[0] is not threading.main_thread()
  assert ticks == 10  # the loop kept running while the token call slept
  assert isinstance(results[0], Exception)


async def test_health_never_fetches_a_token(monkeypatch):
  def boom(self):
    raise AssertionError("health fetched a token")

  monkeypatch.setattr(health_router_mod.UPSOauth, "get_token", boom)
  monkeypatch.setattr(health_router_mod.UPSOauth, "_generate_token", boom)

  class _Client:
    def __init__(self, *a, **k):
      pass

    async def __aenter__(self):
      return self

    async def __aexit__(self, *a):
      return False

    async def head(self, url):
      return type("R", (), {"status_code": 404})()

  monkeypatch.setattr(health_router_mod.httpx, "AsyncClient", _Client)
  body = await health_router_mod.health_check()
  assert body["status"] == "healthy"
