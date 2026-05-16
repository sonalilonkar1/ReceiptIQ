# scripts/eval_sroie_extraction.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tools.vision import extract_fields_from_image
from app.tools.donut import extract_fields_donut
from app.tools.donut import _normalize_date_donut  


def norm_text(s) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 &.'/-]+", "", s)
    return s.strip()


def vendor_match(pred: str, gt: str) -> bool:
    p = norm_text(pred)
    g = norm_text(gt)
    if not p or not g:
        return False
    # substring match either direction (robust to OCR noise)
    return (g in p) or (p in g)


def norm_amount(x) -> float | None:
    if x is None:
        return None
    try:
        return float(str(x).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReceiptIQ extraction on SROIE ground truth.")
    parser.add_argument("--img_dir", type=str, default="data/sroie_100/images")
    parser.add_argument("--lbl_dir", type=str, default="data/sroie_100/labels")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", type=str, default="ocr", choices=["ocr", "donut"])
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    lbl_dir = Path(args.lbl_dir)

    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[: args.limit]
    if not imgs:
        raise SystemExit(f"No images found in {img_dir}. Run download script first.")

    n = 0
    v_ok = 0
    d_ok = 0
    t_ok = 0
    all_ok = 0

    for idx, img_path in enumerate(imgs, start=1):
        label_path = lbl_dir / (img_path.stem + ".json")
        if not label_path.exists():
            continue

        gt = json.load(open(label_path, "r", encoding="utf-8"))
        gt_vendor = gt.get("company")
        gt_date = gt.get("date")
        gt_total = gt.get("total")

        if args.mode == "ocr":
            pred = extract_fields_from_image(str(img_path))
        else:
            pred = extract_fields_donut(image_path=str(img_path), task="sroie")
        pv = pred.get("vendor")
        pd = pred.get("date")
        pt = pred.get("total")

        v_match = vendor_match(pv, gt_vendor)
        pd_norm = _normalize_date_donut(pd) if pd else None
        gt_norm = _normalize_date_donut(gt_date) if gt_date else None
        d_match = (pd_norm == gt_norm) if (pd_norm and gt_norm) else False

        pt_f = norm_amount(pt)
        gt_f = norm_amount(gt_total)
        t_match = (pt_f is not None and gt_f is not None and abs(pt_f - gt_f) <= 0.02)

        v_ok += int(v_match)
        d_ok += int(d_match)
        t_ok += int(t_match)
        all_ok += int(v_match and d_match and t_match)
        n += 1
        if n < 3:
            print("IMG:", img_path.name)
            print("GT :", gt_vendor, gt_date, gt_total)
            print("PR :", pv, pd, pt)
        if idx % 5 == 0:
            print(f"[{idx}/{len(imgs)}] evaluated={n} vendor_ok={v_ok} date_ok={d_ok} total_ok={t_ok}", flush=True)
    print(f"N evaluated: {n}")
    print(f"Vendor match (substring): {v_ok/n:.2%}")
    print(f"Date exact match:         {d_ok/n:.2%}")
    print(f"Total numeric match:      {t_ok/n:.2%}")
    print(f"All three correct:        {all_ok/n:.2%}")


if __name__ == "__main__":
    main()