import argparse
import sys
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lane_mask import lanes_to_mask, overlay_mask
from src.paths import DATA_ROOT, OUTPUTS_DIR
from src.tusimple_dataset import TuSimpleDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--n", type=int, default=4, help="number of samples to save")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "mask_vis")
    args = parser.parse_args()

    ds = TuSimpleDataset()
    args.out.mkdir(parents=True, exist_ok=True)

    for i in range(args.index, min(args.index + args.n, len(ds))):
        rec = ds.samples[i]
        img_path = DATA_ROOT / rec["raw_file"]
        image_bgr = cv2.imread(str(img_path))
        h, w = image_bgr.shape[:2]
        mask = lanes_to_mask(rec["lanes"], rec["h_samples"], h, w)
        vis = overlay_mask(image_bgr, mask)

        out_path = args.out / f"sample_{i:04d}.png"
        cv2.imwrite(str(out_path), vis)
        print(f"saved {out_path} ({img_path.name})")

    print(f"Done. Outputs in {args.out}")


if __name__ == "__main__":
    main()
