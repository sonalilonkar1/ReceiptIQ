# scripts/download_sroie_100.py
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from datasets import load_dataset


def parse_ground_truth(gt: str) -> dict:
    """
    Metric-AI/icdar_sroie stores ground_truth as a stringified Python dict.
    Example: "{'company': '...', 'date': '...', 'total': '...'}"
    """
    if not gt:
        return {}
    try:
        obj = ast.literal_eval(gt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SROIE subset from Hugging Face (Metric-AI/icdar_sroie).")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=str, default="data/sroie_100")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Metric-AI/icdar_sroie", split=args.split)

    n = min(args.n, len(ds))
    for i in range(n):
        ex = ds[i]

        # Save image
        img = ex["image"]
        img_path = img_dir / f"sroie_{args.split}_{i:05d}.jpg"
        img.save(img_path)

        # Parse ground truth dict
        gt = parse_ground_truth(ex.get("ground_truth", ""))
        label = {
            "company": gt.get("company"),
            "date": gt.get("date"),
            "total": gt.get("total"),
            "address": gt.get("address"),
        }

        # Save labels
        label_path = lbl_dir / f"sroie_{args.split}_{i:05d}.json"
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

        if (i + 1) % 25 == 0:
            print(f"Downloaded {i+1}/{n}")

    print(f"\nDone. Saved {n} samples to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()