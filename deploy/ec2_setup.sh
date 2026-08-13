#!/usr/bin/env bash
# ============================================================================
# KL ERP Backend - EC2 one-shot setup / update script (Ubuntu 22.04 / 24.04)
#
# Usage:
#   1. Upload this project to the EC2 instance (git clone / scp / rsync).
#   2. cd into the project directory.
#   3. bash deploy/ec2_setup.sh
#   4. Edit /opt/kl-erp-backend/.env  (GAME_JWT_SECRET, MONGODB_URI, ...)
#   5. sudo systemctl restart kl-erp-backend
#
# The script is idempotent - safe to re-run for every deploy/update.
# ============================================================================
set -euo pipefail

APP_DIR=/opt/kl-erp-backend
APP_USER=ubuntu
SERVICE_NAME=kl-erp-backend
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> [1/6] Installing system packages ..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip build-essential curl rsync

echo "==> [2/6] Syncing app files to ${APP_DIR} ..."
sudo mkdir -p "${APP_DIR}"
# Copy code + model, but never clobber the production .env or live logs.
sudo rsync -a --delete \
    --exclude '.env' \
    --exclude 'logs/' \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude 'extracted/' \
    --exclude '*.zip' \
    "${SRC_DIR}/" "${APP_DIR}/"
sudo chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"

echo "==> [3/6] Creating venv and installing Python deps ..."
sudo -u "${APP_USER}" python3 -m venv "${APP_DIR}/venv"
sudo -u "${APP_USER}" "${APP_DIR}/venv/bin/pip" install --upgrade pip
sudo -u "${APP_USER}" "${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

echo "==> [4/6] Ensuring .env exists ..."
if [ ! -f "${APP_DIR}/.env" ]; then
    sudo -u "${APP_USER}" cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
    echo "    Created ${APP_DIR}/.env from template - EDIT IT before starting."
fi

echo "==> [5/6] Installing systemd unit ..."
sudo cp "${APP_DIR}/deploy/kl-erp-backend.service" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

echo "==> [6/6] (Re)starting ${SERVICE_NAME} ..."
# First start creates the AWS API Gateways; can take ~30-90s.
sudo systemctl restart "${SERVICE_NAME}"

echo ""
echo "============================================================================"
echo " Done. Useful commands:"
echo "   sudo systemctl status ${SERVICE_NAME}"
echo "   journalctl -u ${SERVICE_NAME} -f          # live logs"
echo "   curl http://localhost:8000/               # health check"
echo "============================================================================"
