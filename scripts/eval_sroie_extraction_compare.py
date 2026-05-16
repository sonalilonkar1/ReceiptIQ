#!/usr/bin/env python3
"""
Benchmark ReceiptIQ extraction on SROIE-100 using a "real app" pipeline:

Pipeline (per image):
  1) OCR (Tesseract) -> raw_text
  2) Phi LLM parse from raw_text (extract vendor/date/total)
  3) Donut fallback (only if missing critical fields from Phi OR low confidence)
  4) Merge (fill missing only)

Modes:
  - ocr_phi: Phi parse only (no Donut)
  - donut_only: Donut only
  - real_pipeline: Phi parse + Donut fallback
  - both: run ocr_phi and real_pipeline in the same loop (no double Phi calls)

Outputs:
  - Console summary
  - CSV: outputs/sroie_compare_results.csv
"""

from __future__ import annotations

import sys
import json
import argparse
import re
import time
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tools.vision import extract_fields_from_image
from app.tools.llm_parser import parse_receipt_text_with_llm
from app.tools.donut import extract_fields_donut


# -------------------------
# Helpers: normalization & matching
# -------------------------

def normalize_string(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^\w]", "", str(s)).lower().strip()


def vendor_match(extracted: Optional[str], ground_truth: Optional[str]) -> bool:
    """Robust vendor match: case/whitespace/punctuation insensitive substring match."""
    if not extracted or not ground_truth:
        return False
    e = normalize_string(extracted)
    g = normalize_string(ground_truth)
    return bool(e and g and ((e in g) or (g in e)))

def normalize_date_for_comparison(d: str | None) -> str:
    """Normalize many formats to YYYY-MM-DD."""
    if not d:
        return ""
    s = str(d).strip()

    # Strip time if present: "5/21/2018 13:48" -> "5/21/2018"
    s = s.split(" ")[0].strip()
    s = s.replace(".", "/").replace("\\", "/")

    # Already ISO
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        pass

    # Compact YYYYMMDD
    if re.fullmatch(r"\d{8}", s):
        try:
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            pass

    # Day-first formats (SROIE often DD-MM-YY)
    for fmt in ("%d-%m-%y", "%d/%m/%y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            # handle weird 2-digit-year pivots
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    # Month-first formats
    for fmt in ("%m/%d/%y", "%m-%d-%y", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    # Month names: 14/JUN/2017
    for fmt in ("%d/%b/%Y", "%d/%b/%y", "%d-%b-%Y", "%d-%b-%y"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    return ""


def date_match(extracted: Optional[str], ground_truth: Optional[str]) -> bool:
    if not extracted or not ground_truth:
        return False
    e = normalize_date_for_comparison(extracted)
    g = normalize_date_for_comparison(ground_truth)
    return bool(e and g and e == g)

def _parse_amount(x: Any) -> Optional[float]:
    if x is None:
        return None
    # If already numeric
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # Remove currency words/symbols and parentheses notes
    s = s.replace(",", "")
    s = re.sub(r"(?i)rm|usd|eur|gbp|inr|cad|aud|jpy", "", s)
    s = s.replace("$", "").replace("€", "").replace("£", "")
    # Extract first plausible number (prefer decimal with 2 digits)
    m = re.search(r"(\d+\.\d{2})", s)
    if not m:
        m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def total_match(extracted: Any, ground_truth: Any, tolerance: float = 0.02) -> bool:
    e = _parse_amount(extracted)
    g = _parse_amount(ground_truth)
    if e is None or g is None:
        return False
    try:
        e = float(extracted)
        g = float(ground_truth)
        return abs(e - g) <= tolerance
    except Exception:
        return False


# -------------------------
# Data loader
# -------------------------

def load_sroie_labels(limit: int = 100) -> List[Dict[str, Any]]:
    labels_dir = Path(__file__).parent.parent / "data" / "sroie_100" / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"❌ Labels directory not found: {labels_dir}")

    samples: List[Dict[str, Any]] = []
    for json_file in sorted(labels_dir.glob("*.json"))[:limit]:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # SROIE labels in your dataset use: company/date/total
        gt_vendor = data.get("company") or data.get("store_name") or data.get("merchant")
        gt_date = data.get("date")
        gt_total = data.get("total")

        # coerce total if string
        try:
            gt_total_val = float(str(gt_total).replace(",", "").strip()) if gt_total is not None else None
        except Exception:
            gt_total_val = None

        samples.append({
            "file": json_file.stem,
            "vendor": gt_vendor,
            "date": str(gt_date) if gt_date is not None else None,
            "total": gt_total_val,
        })
    return samples


def resolve_image_path(stem: str) -> Path:
    images_dir = Path(__file__).parent.parent / "data" / "sroie_100" / "images"
    jpg = images_dir / f"{stem}.jpg"
    png = images_dir / f"{stem}.png"
    if jpg.exists():
        return jpg
    if png.exists():
        return png
    return jpg  # default for printing errors


# -------------------------
# Extraction stages (optimized)
# -------------------------

def run_ocr(image_path: str) -> Tuple[str, float]:
    """Return raw_text, latency_ms."""
    t0 = time.perf_counter()
    ocr = extract_fields_from_image(image_path)
    raw = ocr.get("raw_text", "") or ""
    ms = (time.perf_counter() - t0) * 1000
    return raw, ms


def run_phi_parse(raw_text: str) -> Tuple[Dict[str, Any], float]:
    """Return parsed fields dict + latency_ms."""
    t0 = time.perf_counter()
    llm = parse_receipt_text_with_llm(raw_text) or {}
    ms = (time.perf_counter() - t0) * 1000

    # Your llm_parser returns nested dicts like {"vendor":{"value":..,"confidence":..}}
    def gv(k: str):
        return (llm.get(k) or {}).get("value")

    def gc(k: str):
        try:
            return float((llm.get(k) or {}).get("confidence", 0))
        except Exception:
            return 0.0

    out = {
        "vendor": gv("vendor"),
        "date": gv("date"),
        "total": gv("total"),
        "conf_vendor": gc("vendor"),
        "conf_date": gc("date"),
        "conf_total": gc("total"),
    }
    return out, ms


def run_donut(image_path: str) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    d = extract_fields_donut(image_path=image_path, task="sroie") or {}
    ms = (time.perf_counter() - t0) * 1000
    return d, ms


def merge_fill_missing(base: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Fill only missing critical fields in base using fallback."""
    out = dict(base)
    for k in ("vendor", "date", "total"):
        if (out.get(k) in (None, "", 0)) and fallback.get(k) not in (None, "", 0):
            out[k] = fallback[k]
    return out


# -------------------------
# Main benchmark
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="SROIE benchmark: Phi vs Donut fallback (real pipeline).")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", choices=["ocr_phi", "donut_only", "real_pipeline", "both"], default="both")
    parser.add_argument("--conf_threshold", type=float, default=0.60, help="Trigger Donut if Phi confidence below this")
    parser.add_argument("--save_csv", action="store_true", help="Save outputs/sroie_compare_results.csv")
    parser.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help="Optional output CSV filename (e.g., outputs/sroie_real_pipeline.csv). If not set, uses a default name."
    )
    args = parser.parse_args()

    print(f"📊 Loading SROIE-100 samples (limit={args.limit})...")
    samples = load_sroie_labels(args.limit)
    print(f"   Loaded {len(samples)} samples\n")

    results: List[Dict[str, Any]] = []

    # aggregate timing + coverage
    ocr_ms_all = []
    phi_ms_all = []
    donut_ms_all = []

    donut_trigger_count = 0

    for i, gt in enumerate(samples, 1):
        stem = gt["file"]
        img_path = resolve_image_path(stem)
        if not img_path.exists():
            print(f"⚠ {i}/{len(samples)}: Image not found: {img_path}")
            continue

        print(f"🔍 {i}/{len(samples)}: Processing {stem}...", end=" ", flush=True)

        row: Dict[str, Any] = {
            "file": stem,
            "gt_vendor": gt.get("vendor"),
            "gt_date": gt.get("date"),
            "gt_total": gt.get("total"),
        }

        # Stage 1 OCR once
        raw_text, ocr_ms = run_ocr(str(img_path))
        ocr_ms_all.append(ocr_ms)

        # Mode: donut_only
        if args.mode == "donut_only":
            donut_out, donut_ms = run_donut(str(img_path))
            donut_ms_all.append(donut_ms)

            row.update({
                "pred_vendor": donut_out.get("vendor"),
                "pred_date": donut_out.get("date"),
                "pred_total": donut_out.get("total"),
                "source": "donut",
                "latency_ocr_ms": round(ocr_ms, 2),
                "latency_phi_ms": None,
                "latency_donut_ms": round(donut_ms, 2),
            })

        else:
            # Stage 2 Phi parse once
            phi_out, phi_ms = run_phi_parse(raw_text)
            phi_ms_all.append(phi_ms)

            base_pred = {
                "vendor": phi_out.get("vendor"),
                "date": phi_out.get("date"),
                "total": phi_out.get("total"),
                "source": "phi",
            }

            # Mode: ocr_phi (no donut)
            if args.mode == "ocr_phi":
                final_pred = base_pred
                row.update({
                    "pred_vendor": final_pred.get("vendor"),
                    "pred_date": final_pred.get("date"),
                    "pred_total": final_pred.get("total"),
                    "source": "phi",
                    "phi_conf_vendor": phi_out.get("conf_vendor"),
                    "phi_conf_date": phi_out.get("conf_date"),
                    "phi_conf_total": phi_out.get("conf_total"),
                    "latency_ocr_ms": round(ocr_ms, 2),
                    "latency_phi_ms": round(phi_ms, 2),
                    "latency_donut_ms": None,
                })

            else:
                # Mode: real_pipeline or both
                missing = [k for k in ("vendor", "date", "total") if not base_pred.get(k)]
                low_conf = [k for k in ("vendor", "date", "total")
                            if phi_out.get(f"conf_{k}", 1.0) < args.conf_threshold]

                donut_used = False
                donut_ms = None

                if missing or low_conf:
                    donut_trigger_count += 1
                    donut_out, donut_ms_val = run_donut(str(img_path))
                    donut_used = True
                    donut_ms = donut_ms_val
                    donut_ms_all.append(donut_ms_val)

                    final_pred = merge_fill_missing(base_pred, donut_out)
                    final_pred["source"] = "phi+donut"
                else:
                    final_pred = base_pred

                row.update({
                    "pred_vendor": final_pred.get("vendor"),
                    "pred_date": final_pred.get("date"),
                    "pred_total": final_pred.get("total"),
                    "source": final_pred.get("source"),
                    "phi_conf_vendor": phi_out.get("conf_vendor"),
                    "phi_conf_date": phi_out.get("conf_date"),
                    "phi_conf_total": phi_out.get("conf_total"),
                    "donut_used": donut_used,
                    "latency_ocr_ms": round(ocr_ms, 2),
                    "latency_phi_ms": round(phi_ms, 2),
                    "latency_donut_ms": round(donut_ms, 2) if donut_ms is not None else None,
                })

        # Matches
        row["vendor_match"] = vendor_match(row.get("pred_vendor"), gt.get("vendor"))
        row["date_match"] = date_match(row.get("pred_date"), gt.get("date"))
        row["total_match"] = total_match(row.get("pred_total"), gt.get("total"))
        row["all3_match"] = bool(row["vendor_match"] and row["date_match"] and row["total_match"])

        results.append(row)
        print("✅")

    # -------------------------
    # Summary
    # -------------------------
    n = len(results)
    print("\n" + "=" * 80)
    print("📈 Results Summary")
    print("=" * 80)
    print(f"N evaluated: {n}")

    if n == 0:
        print("No samples evaluated (check image/label paths).")
        return

    vendor_acc = sum(r["vendor_match"] for r in results) / n
    date_acc = sum(r["date_match"] for r in results) / n
    total_acc = sum(r["total_match"] for r in results) / n
    all3_acc = sum(r["all3_match"] for r in results) / n

    # Coverage
    cov_vendor = sum(1 for r in results if r.get("pred_vendor")) / n
    cov_date = sum(1 for r in results if r.get("pred_date")) / n
    cov_total = sum(1 for r in results if r.get("pred_total") not in (None, "", 0)) / n

    print(f"\nAccuracy:")
    print(f"  Vendor: {vendor_acc*100:6.2f}%")
    print(f"  Date:   {date_acc*100:6.2f}%")
    print(f"  Total:  {total_acc*100:6.2f}%")
    print(f"  All-3:  {all3_acc*100:6.2f}%")

    print(f"\nCoverage (non-null predictions):")
    print(f"  Vendor filled: {cov_vendor*100:6.2f}%")
    print(f"  Date filled:   {cov_date*100:6.2f}%")
    print(f"  Total filled:  {cov_total*100:6.2f}%")

    # Timing
    def avg(xs):
        return sum(xs)/len(xs) if xs else 0

    print(f"\nLatency (avg ms):")
    print(f"  OCR:   {avg(ocr_ms_all):8.2f} ms")
    if phi_ms_all:
        print(f"  Phi:   {avg(phi_ms_all):8.2f} ms")
    if donut_ms_all:
        print(f"  Donut: {avg(donut_ms_all):8.2f} ms  (called on {donut_trigger_count}/{n} images)")

    # Save CSV
    if args.save_csv:
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        if args.csv_name:
            csv_path = Path(args.csv_name)
            # If user passed relative path, keep it under repo root
            if not csv_path.is_absolute():
                csv_path = (Path(__file__).parent.parent / csv_path).resolve()
        else:
            # default unique name by mode
            csv_path = output_dir / f"sroie_compare_results_{args.mode}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        print(f"\n💾 Saved CSV to: {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()