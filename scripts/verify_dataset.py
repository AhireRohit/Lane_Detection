import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLIPS_DIR, DATA_ROOT, LABEL_JSON


def main():
    assert DATA_ROOT.is_dir(), f"Missing data root: {DATA_ROOT}"
    assert LABEL_JSON.is_file(), f"Missing labels: {LABEL_JSON}"
    assert CLIPS_DIR.is_dir(), f"Missing clips: {CLIPS_DIR}"

    n = 0
    missing = []
    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            img = DATA_ROOT / rec["raw_file"]
            if not img.is_file():
                missing.append(rec["raw_file"])

    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Annotations: {n}")
    print(f"Missing images: {len(missing)}")
    if missing[:5]:
        print("Examples:", missing[:5])



if __name__ == "__main__":
    main()
