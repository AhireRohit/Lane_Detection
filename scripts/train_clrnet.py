import argparse
import json
import random
import sys
import time
from pathlib import Path

import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import segmentation_metrics
from src.paths import CHECKPOINTS_DIR, OUTPUTS_DIR
from src.tusimple_dataset import TuSimpleDataset
from src.clrnet_seg import CLRNetSeg


def get_transforms(height, width, train):
    if train:
        return A.Compose(
            [
                A.Resize(height, width),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
    return A.Compose([A.Resize(height, width)])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    totals = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        m = segmentation_metrics(logits, masks)
        for k in totals:
            totals[k] += m[k]
    n = max(len(loader), 1)
    return total_loss / n, {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base", type=int, default=32, help="UNet channel base")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    full_ds = TuSimpleDataset()
    n = len(full_ds)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = TuSimpleDataset(transform=get_transforms(args.height, args.width, True))
    val_ds = TuSimpleDataset(transform=get_transforms(args.height, args.width, False))
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CLRNetSeg(base=args.base).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = CHECKPOINTS_DIR / "clrnet_best.pt"
    history = []
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
            "sec": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"iou {val_metrics['iou']:.4f} f1 {val_metrics['f1']:.4f} | "
            f"{row['sec']}s"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": best_iou,
                    "config": vars(args),
                },
                ckpt_path,
            )

    log_path = OUTPUTS_DIR / "clrnet_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"best val IoU: {best_iou:.4f}")
    print(f"checkpoint: {ckpt_path}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
