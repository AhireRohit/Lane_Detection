import argparse
import json
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lane_mask import overlay_mask
from src.metrics import segmentation_metrics
from src.paths import CHECKPOINTS_DIR, OUTPUTS_DIR
from src.tusimple_dataset import TuSimpleDataset
from src.unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "unet_best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    height = cfg.get("height", args.height)
    width = cfg.get("width", args.width)
    base = cfg.get("base", 32)

    transform = A.Compose([A.Resize(height, width)])
    ds = TuSimpleDataset(transform=transform)

    n = len(ds)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    rng = __import__("random").Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val_idx = indices[n_train:]

    loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    totals = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    latencies = []
    vis_dir = OUTPUTS_DIR / "unet_preds"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_saved = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) / images.size(0))

            m = segmentation_metrics(logits, masks)
            for k in totals:
                totals[k] += m[k]

            if vis_saved < args.save_vis:
                probs = torch.sigmoid(logits).cpu().numpy()
                imgs = images.cpu().numpy()
                gts = masks.cpu().numpy()
                for i in range(min(images.size(0), args.save_vis - vis_saved)):
                    img = (imgs[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    pred_mask = (probs[i, 0] > 0.5).astype(np.uint8)
                    gt_mask = (gts[i, 0] > 0.5).astype(np.uint8)
                    vis = np.hstack(
                        [
                            overlay_mask(img_bgr, gt_mask, color=(0, 255, 0)),
                            overlay_mask(img_bgr, pred_mask, color=(0, 0, 255)),
                        ]
                    )
                    out = vis_dir / f"pred_{vis_saved:03d}.png"
                    cv2.imwrite(str(out), vis)
                    vis_saved += 1

    n_batches = max(len(loader), 1)
    metrics = {k: v / n_batches for k, v in totals.items()}
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    fps = 1.0 / avg_lat if avg_lat > 0 else 0.0
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    report = {
        **metrics,
        "latency_ms": round(avg_lat * 1000, 2),
        "fps": round(fps, 1),
        "params": param_count,
        "size_mb": round(size_mb, 2),
        "checkpoint": str(args.checkpoint),
        "val_images": len(val_idx),
    }

    out_json = OUTPUTS_DIR / "unet_eval.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"predictions: {vis_dir}")
    print(f"report: {out_json}")


if __name__ == "__main__":
    main()
