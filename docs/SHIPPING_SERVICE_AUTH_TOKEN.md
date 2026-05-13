# SHIPPING_SERVICE_AUTH_TOKEN

Bearer token for HTTP auth between `core/admin` and `ups-rate-service`. Both
services must hold the same value per environment. The token is independent
of the carrier credentials (`FEDEX_CLIENT_ID`, `UPS_CLIENT_ID`, etc.) —
those authenticate to FedEx/UPS/USPS; this one authenticates Crystal-to-Crystal.

## Where it lives

| Service          | Env var name                  | Notes                            |
|------------------|-------------------------------|----------------------------------|
| ups-rate-service | `AUTH_TOKEN`                  | Validated on every inbound request |
| core/admin       | `SHIPPING_SERVICE_AUTH_TOKEN` | Sent as `Authorization: Bearer <token>` to ups-rate-service |

Values MUST match within an environment. Values SHOULD differ across
environments (staging value != production value).

## Generate a new token

```bash
openssl rand -base64 48 | tr -d '/+=' | cut -c1-56
```

Produces a 56-char alphanumeric string. Length and charset chosen to match
the existing staging token (`A7fK9xQ2mZr8Lp4VwT1cY6nB3dE5uH0sJqRkXoCzN8gM2tP7yWvFhD9aU`).

## Install

Set both env vars in the deployment manifests for the target environment.
Production example (replace `<TOKEN>` with the generated value):

```bash
# ups-rate-service
kubectl -n default set env deployment/prd-ups-microservice AUTH_TOKEN='<TOKEN>'

# core/admin
kubectl -n default set env deployment/prd-core-admin SHIPPING_SERVICE_AUTH_TOKEN='<TOKEN>'
```

If your deployment manifests are managed via CTO.ai / k8s manifests in a repo
rather than via `kubectl set env`, update those manifests and apply. The
token should be stored as a Kubernetes Secret in the long run, not inline.

## Verify

After deploy + pod rollout, from any pod or your laptop with cluster access:

```bash
TOKEN='<the value just set>'
PAYLOAD='{"sender_zip":"98101","sender_country":"US","zip":"10001","country":"US","weight":2.5}'

# Replace host with the appropriate environment URL
curl -s -X POST 'https://ups-microservice.stg.crystalcommerce.info/ups/rates' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD" | head -c 300
```

- 200 + a `quotes` array → token is good
- 401 → token mismatch between caller and service (or not yet rolled out to a pod)

## Rotate

Both services have to pick up the new value in close sequence or there's a
window of 401s during which checkout fails.

**Preferred (zero downtime):** make ups-rate-service accept both old and new
tokens temporarily. Requires a code change in ups-rate-service to read a
list rather than a single value, deploy that, then flip core/admin to the
new token, then remove the old from ups-rate-service. No checkout
disruption.

**Pragmatic (brief disruption):** flip both env vars simultaneously and
accept ~30-60s of 401s during the rolling deploy. Acceptable during a
low-traffic window.

## Caveats

- **Never paste the token into Slack, chat, commits, or screenshots.** If
  you do, treat it as compromised and rotate.
- **One person rotates at a time.** Coordinate in the team channel before
  starting — a half-rotated cluster is the failure mode that causes
  checkout 401s.
- **Token is independent of FedEx/UPS/USPS credentials.** Rotating this
  token does NOT require any FedEx Developer Portal action.

## Related

- FedEx migration context: `core/admin/tasks/fedex-migration-rebase-plan.md`
- Carrier credential setup (FEDEX_CLIENT_ID, etc.): see `.env.example`
  in this repo for the full env-var list.
