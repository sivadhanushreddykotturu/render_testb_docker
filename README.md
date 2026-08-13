# render_testb_docker

FastAPI backend for the KL University ERP student app — timetable, attendance,
grades, seating plan, auto-feedback, and break-time game leaderboards.

Runs on AWS EC2 and rotates egress IPs through **AWS API Gateway**
(`requests-ip-rotator`) so the university portal never rate-limits a single IP.

## Deploy

See **[DEPLOYMENT.md](DEPLOYMENT.md)** — one-shot setup on EC2 with Docker
or systemd:

```bash
git clone https://github.com/sivadhanushreddykotturu/render_testb_docker.git
cd render_testb_docker
cp .env.example .env   # edit secrets
docker build -t kl-erp-backend . && docker run -d --name kl-erp-backend \
  --network host --env-file .env --restart unless-stopped kl-erp-backend
```
