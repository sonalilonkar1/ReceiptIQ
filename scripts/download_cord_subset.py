from pathlib import Path
import json
from datasets import load_dataset

OUT_DIR = Path("data/cord_100")
N = 100
SPLIT = "train"  # can also try "validation" if available

def main():
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    ds = load_dataset("naver-clova-ix/cord-v2", split=SPLIT)

    for i in range(min(N, len(ds))):
        ex = ds[i]

        # Save image
        img = ex["image"]
        img_path = OUT_DIR / "images" / f"cord_{SPLIT}_{i:05d}.png"
        img.save(img_path)

        # Save ground truth (usually JSON-in-string for Donut-style datasets)
        gt = ex.get("ground_truth", None)
        label_path = OUT_DIR / "labels" / f"cord_{SPLIT}_{i:05d}.json"
        with open(label_path, "w", encoding="utf-8") as f:
            if isinstance(gt, str):
                f.write(gt)
            else:
                json.dump(gt, f, ensure_ascii=False, indent=2)

    print(f"Saved {min(N, len(ds))} samples to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()