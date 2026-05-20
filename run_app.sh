#!/bin/bash
set -euxo pipefail

cd /home/atml_team066/Lane_detection

export PYTHONNOUSERSITE=1
export JOB_ENV=$(mktemp -d /tmp/lanedet_app_env_XXXXXX)
export PYTHONPATH="$JOB_ENV:${PYTHONPATH:-}"
export PATH="$JOB_ENV/bin:$PATH"

python -m pip install --no-cache-dir --target "$JOB_ENV" \
  numpy==1.26.4 \
  opencv-python-headless==4.10.0.84 \
  albumentations==1.4.8 \
  albucore==0.0.9 \
  gradio==3.50.2 \
  huggingface_hub==0.25.2 \
  matplotlib \
  tqdm

python app/gradio_app.py