#!/bin/bash
set -euxo pipefail

cd /home/atml_team066/Lane_detection

python -m pip uninstall -y albumentations albucore opencv-python opencv-python-headless numpy || true

python -m pip install --user --no-cache-dir --force-reinstall \
  numpy==1.26.4 \
  opencv-python-headless==4.10.0.84 \
  albumentations==1.4.8 \
  albucore==0.0.9 \
  tqdm matplotlib

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import cv2, numpy, albumentations; print('cv2', cv2.__version__); print('numpy', numpy.__version__); print('albumentations', albumentations.__version__)"

python scripts/train_scnn.py --epochs 30 --batch-size 8
python scripts/eval_scnn.py