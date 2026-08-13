# KL ERP Backend — AWS Deployment & Ops Guide

FastAPI backend that proxies `https://newerp.kluniversity.in` through **AWS API Gateway pass-through proxies** (`requests-ip-rotator`) for IP rotation, served by **Gunicorn + Uvicorn workers** on **EC2 (ap-south-1/2)**.

## Architecture

```
client app ──► EC2 :8000 (Gunicorn → Uvicorn workers → FastAPI)
                     │
                     │  ApiGatewayTransport (gateway_proxy.py, async httpx)
                     │  per request: random endpoint + random X-Forwarded-For
                     ▼
        {api-id}.execute-api.{region}.amazonaws.com/ProxyStage/<path>
        (REST API, {proxy+} HTTP_PROXY integration, ap-south-1 + ap-south-2)
                     │  egress IP rotates per request (AWS IP pool)
                     ▼
              newerp.kluniversity.in
```

**Lifecycle rule:** `ApiGateway.start()` runs **once** at boot — in the Gunicorn
master (`gunicorn_conf.py: on_starting`) and again in each worker's FastAPI
`lifespan` (which finds the already-existing gateways by name, so it takes
~1s). It is **never** run per-request. Each request just picks a random
endpoint from the in-memory list and rewrites the URL — no AWS API calls on
the hot path.

`start()` reuses existing gateways by name (`<site> - IP Rotate API`), so
restarts/redeploys do **not** recreate infrastructure. Gateways are left in
place on shutdown by default (`DELETE_GATEWAYS_ON_SHUTDOWN=false`); idle
REST APIs cost ~$0 (pay per request, first 1M req/region free tier).

---

## Step 1 — One-time EC2 provisioning

1. **Launch instance:** Ubuntu 24.04 LTS, `t3.small` (or `t3.micro`), region
   `ap-south-1` (Mumbai) or `ap-south-2` (Hyderabad).
2. **Security group:** inbound TCP `8000` from your client app network
   (and `22` from your IP). Outbound: all (default).
3. **IAM instance role (recommended over access keys):** IAM → Roles →
   create role for EC2 with the managed policy
   **`AmazonAPIGatewayAdministrator`** → attach to the instance
   (EC2 → instance → Actions → Security → Modify IAM role).
   boto3 picks it up automatically — no keys in `.env`.

## Step 2 — Get the code on the instance

```bash
ssh ubuntu@<ec2-public-ip>
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/sivadhanushreddykotturu/render_testb_docker.git
cd render_testb_docker
cp .env.example .env
nano .env          # set GAME_JWT_SECRET (required), MONGODB_URI (optional)
```

## Step 3 — Run it (pick ONE)

### Option A — Docker (recommended)

```bash
# Install Docker
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu && newgrp docker

# Build + run
docker build -t kl-erp-backend .
docker run -d --name kl-erp-backend --network host \
    --env-file .env --restart unless-stopped \
    -v "$(pwd)/logs:/app/logs" \
    kl-erp-backend
```

> **`--network host` matters:** boto3 fetches IAM-role credentials from the
> instance metadata service (169.254.169.254). With default bridge
> networking, IMDSv2 rejects container requests (hop limit 1). If you prefer
> bridge networking (`-p 8000:8000`), raise the hop limit instead:
> `aws ec2 modify-instance-metadata-options --instance-id <id> --http-put-response-hop-limit 2 --http-tokens required`
> — or just put AWS keys in `.env`.

Or with compose (same thing): `docker compose up -d --build`

### Option B — systemd (no Docker)

```bash
bash deploy/ec2_setup.sh    # installs venv, deps, systemd unit; restarts service
```

## Step 4 — Verify

```bash
curl http://localhost:8000/
docker logs -f kl-erp-backend                       # Docker
journalctl -u kl-erp-backend -f                     # systemd
```

Expected health response:

```json
{"message":"Backend running high-speed concurrent loops ✅","status":"healthy",
 "gateway":{"regions":["ap-south-1","ap-south-2"],"endpoints_live":2}}
```

First boot takes **~30–90s** (creates the API Gateways in AWS). Then point
your client app at `http://<ec2-public-ip>:8000`.

---

## Day-to-day ops

| Task | Docker | systemd |
|---|---|---|
| Live logs | `docker logs -f kl-erp-backend` | `journalctl -u kl-erp-backend -f` |
| Restart | `docker restart kl-erp-backend` | `sudo systemctl restart kl-erp-backend` |
| Deploy update | `git pull && docker build -t kl-erp-backend . && docker rm -f kl-erp-backend && <run cmd again>` | `git pull && bash deploy/ec2_setup.sh` |
| Change env | edit `.env` → `docker rm -f kl-erp-backend && <run cmd>` | edit `/opt/kl-erp-backend/.env` → `sudo systemctl restart kl-erp-backend` |
| Force-recreate gateways | delete APIs named `…IP Rotate API` in API Gateway console → restart | same |
| Tear down gateways | set `DELETE_GATEWAYS_ON_SHUTDOWN=true` in `.env`, stop container/service, set it back | same |

## Config reference (`.env`)

| Var | Default | Purpose |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | *(unset)* | Only if no IAM instance role. Never set them empty. |
| `AWS_GATEWAY_REGIONS` | `ap-south-1,ap-south-2` | Regions to build proxy gateways in. More regions = bigger IP pool. |
| `DELETE_GATEWAYS_ON_SHUTDOWN` | `false` | Keep (`false`, fast restarts) or delete (`true`) gateways on stop. |
| `PORT` | `8000` | Bind port. |
| `WEB_CONCURRENCY` | `2` | Gunicorn Uvicorn workers. |
| `ERP_BASE_URL` | `https://newerp.kluniversity.in` | Target portal. |
| `GAME_JWT_SECRET` | *(change me)* | HMAC secret for game tokens. **Change it.** |
| `MONGODB_URI` | *(empty = in-memory)* | Persistent leaderboard store. |
| `FEEDBACK_ALLOWED_USERS` | `2400032717` | CSV allow-list for `/auto-feedback`. |

## Notes & gotchas

- **TLS verification is ON** (`verify=True`): traffic terminates at AWS with
  a valid ACM cert. The old `verify=False` only existed for the residential
  MITM proxy.
- **`X-Forwarded-For` is randomised per request** and passed through the
  gateway's `X-My-X-Forwarded-For` mapping, so the ERP never sees the real
  EC2 IP.
- **Costs:** API Gateway REST ≈ $3.50/M requests after the 1M/region free
  tier; EC2 egress ~$0.09/GB. Watch with AWS Cost Explorer if traffic grows.
- **429 monitoring:** grep logs for `[RATE_LIMIT]` — each hit is tagged with
  the egress gateway host.
