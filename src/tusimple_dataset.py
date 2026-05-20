import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.lane_mask import lanes_to_mask
from src.paths import DATA_ROOT, LABEL_JSON


class TuSimpleDataset(Dataset):
    def __init__(self, label_json=LABEL_JSON, root=DATA_ROOT, transform=None, line_thickness=5):
        self.root = Path(root)
        self.transform = transform
        self.line_thickness = line_thickness
        self.samples = []
        with open(label_json, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.samples.append(
                    {
                        "raw_file": rec["raw_file"],
                        "lanes": rec["lanes"],
                        "h_samples": rec["h_samples"],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_path = self.root / rec["raw_file"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mask = lanes_to_mask(
            rec["lanes"],
            rec["h_samples"],
            h,
            w,
            line_thickness=self.line_thickness,
        )

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return {
            "image": image_t,
            "mask": mask_t,
            "path": str(img_path),
        }
