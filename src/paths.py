from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "train_subset_3626"
LABEL_JSON = DATA_ROOT / "label_data_subset_3626.json"
CLIPS_DIR = DATA_ROOT / "clips"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
APP_DIR = PROJECT_ROOT / "app"

for d in (CHECKPOINTS_DIR, OUTPUTS_DIR, APP_DIR):
    d.mkdir(parents=True, exist_ok=True)
