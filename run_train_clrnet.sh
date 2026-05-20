#!/bin/bash
set -euxo pipefail

cd /home/atml_team066/Lane_detection

export JOB_ENV=$(mktemp -d /tmp/lanedet_clrnet_env_XXXXXX)
export PYTHONPATH="$JOB_ENV:${PYTHONPATH:-}"
export PATH="$JOB_ENV/bin:$PATH"

python -m pip install --no-cache-dir --target "$JOB_ENV" \
  numpy==1.26.4 \
  opencv-python-headless==4.10.0.84 \
  albumentations==1.4.8 \
  albucore==0.0.9 \
  tqdm \
  matplotlib

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import cv2, numpy, albumentations; print('cv2', cv2.__version__); print('numpy', numpy.__version__); print('albumentations', albumentations.__version__)"

python scripts/verify_dataset.py

python scripts/train_clrnet.py --epochs 30 --batch-size 8
python scripts/eval_clrnet.py