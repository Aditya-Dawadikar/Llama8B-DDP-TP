#!/usr/bin/env bash
# Set up a Python environment and install dependencies for LLaMA fine‑tuning.
#
# This script is idempotent and can be run multiple times.  It installs
# system dependencies via apt, creates a Python virtual environment in
# `.venv` and installs Python packages from `requirements.txt`.  CUDA and
# NVIDIA drivers must be installed separately (see comments below).

set -euo pipefail

echo "[setup_env] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git wget build-essential

# CUDA installation is environment specific.  See https://developer.nvidia.com/cuda-downloads
echo "[setup_env] Note: Install CUDA toolkit and NVIDIA drivers appropriate for your GPUs before running training."

VENV_DIR="$(pwd)/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup_env] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "[setup_env] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[setup_env] Installation complete.  Activate the environment with:\n  source $VENV_DIR/bin/activate"