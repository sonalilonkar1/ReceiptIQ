#!/usr/bin/env python3
"""
SROIE extraction benchmark for ReceiptIQ.

Modes:
  - ocr_phi      : OCR-only baseline, with Phi called ONLY if OCR is missing vendor/date/total
  - donut_only   : Donut-only baseline (no OCR run)
  - real_pipeline: OCR -> (Phi only if missing) -> Donut fallback (only if still missing OR Phi confidence below threshold)
  - both         : runs ocr_phi then real_pipeline sequentially

Outputs:
  - Prints accuracy (vendor/date/total/all-3)
  - Prints coverage (non-null predictions)
  - Prints latency averages
  - Optional CSV saving via --save_csv and --csv_name

This version is optimized to avoid slow Phi calls when OCR already extracted the needed fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.vision import extract_fields_from_image  # OCR + regex extraction (returns vendor/date/total/raw_text)
from app.tools.donut import extract_fields_donut        # Donut extraction (returns vendor/date/total)
from app.tools.llm_parser import parse_receipt_text_with_llm  # Phi parsing from OCR text (may be slow)


# ----------------------------
# Normalization & matching
# ----------------------------

def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 &.'/-]+", "", s)
    return s.strip()

def vendor_match(pred: Any, gt: Any) -> bool:
    p = norm_text(pred)
    g = norm_text(gt)
    if not p or not g:
        return False
    return (g in p) or (p in g)

def _normalize_date(s: Any) -> str:
    """Normalize a date into YYYY-MM-DD if possible."""
    if not s:
        return ""
    t = str(s).strip()
    t = t.split(" ")[0].strip()
    t = t.replace(".", "/").replace("\\", "/")
    # try known formats
    fmts = [
        "%Y-%m-%d",
        "%d-%m-%y", "%d/%m/%y",
        "%d-%m-%Y", "%d/%m/%Y",
        "%m/%d/%y", "%m-%d-%y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%d/%b/%Y", "%d/%b/%y",
        "%d-%b-%Y", "%d-%b-%y",
    ]
    # convert separators to consistent style for parsing
    t_dash = t.replace("/", "-")
    for fmt in fmts:
        try:
            dt = datetime.strptime(t, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
        try:
            dt = datetime.strptime(t_dash, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    # YYYYMMDD
    if re.fullmatch(r"\d{8}", t):
        try:
            dt = datetime.strptime(t, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""

def date_match(pred: Any, gt: Any) -> bool:
    p = _normalize_date(pred)
    g = _normalize_date(gt)
    return bool(p and g and p == g)

def _parse_amount(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # remove currency tokens and spaces
    s = s.replace(",", "")
    s = re.sub(r"(?i)\b(rm|usd|eur|gbp|inr|cad|aud|jpy)\b", "", s)
    s = s.replace("$", "").replace("€", "").replace("£", "")
    # prefer decimal amounts (2 decimals, then 1 decimal, then any number)
    m = re.search(r"(\d+\.\d{2})", s)  # e.g., 22.90
    if not m:
        m = re.search(r"(\d+\.\d{1})", s)  # e.g., 22.5
    if not m:
        m = re.search(r"(\d+)", s)  # e.g., 22
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def total_match(pred: Any, gt: Any, tolerance: float = 0.02) -> bool:
    p = _parse_amount(pred)
    g = _parse_amount(gt)
    if p is None or g is None:
        return False
    return abs(p - g) <= tolerance


# ----------------------------
# SROIE loader
# ----------------------------

def load_sroie_samples(limit: int) -> List[Dict[str, Any]]:
    labels_dir = PROJECT_ROOT / "data" / "sroie_100" / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"❌ Labels directory not found: {labels_dir}")

    samples: List[Dict[str, Any]] = []
    for jf in sorted(labels_dir.glob("*.json"))[:limit]:
        d = json.loads(jf.read_text(encoding="utf-8"))
        gt_vendor = d.get("company") or d.get("store_name") or d.get("merchant")
        gt_date = d.get("date")
        gt_total = d.get("total")
        try:
            gt_total_val = float(str(gt_total).replace(",", "").strip()) if gt_total is not None else None
        except Exception:
            gt_total_val = None
        samples.append({
            "stem": jf.stem,
            "vendor": gt_vendor,
            "date": str(gt_date) if gt_date is not None else None,
            "total": gt_total_val,
        })
    return samples

def resolve_image(stem: str) -> Path:
    img_dir = PROJECT_ROOT / "data" / "sroie_100" / "images"
    jpg = img_dir / f"{stem}.jpg"
    png = img_dir / f"{stem}.png"
    if jpg.exists():
        return jpg
    if png.exists():
        return png
    return jpg


# ----------------------------
# Phi parsing helpers
# ----------------------------

def _extract_llm_value(llm: Any, field: str) -> Tuple[Optional[Any], Optional[float]]:
    """
    Supports both shapes:
      - {"field": {"value": ..., "confidence": ...}}
      - {"field": ...}
    """
    if llm is None:
        return None, None
    try:
        if isinstance(llm.get(field), dict):
            v = llm[field].get("value")
            c = llm[field].get("confidence")
            try:
                c = float(c) if c is not None else None
            except Exception:
                c = None
            return v, c
        if field in llm:
            return llm.get(field), None
    except Exception:
        pass
    return None, None

def _jsonish_to_dict(text: str) -> Dict[str, Any]:
    """
    Extract a JSON-like object from a model response.
    Handles JSON wrapped in markdown/backticks, trailing commas, // comments,
    and extra text before/after the JSON.
    """
    if not text:
        return {}
    t = text.strip()

    # Grab first {...} block
    m = re.search(r"\{.*?\}", t, flags=re.DOTALL)
    if not m:
        return {}

    j = m.group(0)

    # Remove // comments
    j = re.sub(r"//.*?$", "", j, flags=re.MULTILINE)

    # Remove trailing commas before } or ]
    j = re.sub(r",\s*([}\]])", r"\1", j)

    # Normalize smart quotes just in case
    j = j.replace("“", "\"").replace("”", "\"").replace("’", "'")

    try:
        return json.loads(j)
    except Exception:
        return {}

def _regex_fallback_from_text(text: str) -> Dict[str, Any]:
    """Fallback extraction when model JSON is malformed."""
    out: Dict[str, Any] = {}
    if not text:
        return out

    dm = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
    if dm:
        out["date"] = dm.group(1)

    tm = re.search(r"(?i)\btotal\b[^\d]{0,20}(\d+[.,]\d{2})", text)
    if tm:
        out["total"] = tm.group(1)
    else:
        tm2 = re.search(r"(\d+[.,]\d{2})", text)
        if tm2:
            out["total"] = tm2.group(1)

    return out

def parse_with_phi(raw_text: str) -> Tuple[Dict[str, Any], float]:
    """
    Call parse_receipt_text_with_llm(raw_text) and normalize into flat dict.

    In your repo, parse_receipt_text_with_llm may return either:
      - a dict (preferred), OR
      - a raw string response from Ollama.

    This function handles both.
    """
    t0 = time.perf_counter()
    llm_raw = parse_receipt_text_with_llm(raw_text)
    ms = (time.perf_counter() - t0) * 1000.0

    # Case A: already structured dict
    if isinstance(llm_raw, dict):
        llm = llm_raw
        v, cv = _extract_llm_value(llm, "vendor")
        d, cd = _extract_llm_value(llm, "date")
        tot, ct = _extract_llm_value(llm, "total")

        if v is None:
            v, _ = _extract_llm_value(llm, "company")
        if tot is None:
            tot, _ = _extract_llm_value(llm, "amount")

        return {
            "vendor": v,
            "date": d,
            "total": tot,
            "conf_vendor": cv,
            "conf_date": cd,
            "conf_total": ct,
        }, ms

    # Case B: raw text (common with Ollama)
    text = str(llm_raw or "")
    data = _jsonish_to_dict(text)
    if not data:
        data = _regex_fallback_from_text(text)

    v = data.get("vendor") or data.get("company") or data.get("merchant")
    d = data.get("date")
    tot = data.get("total")

    return {
        "vendor": v,
        "date": d,
        "total": tot,
        "conf_vendor": None,
        "conf_date": None,
        "conf_total": None,
    }, ms


# ----------------------------
# Benchmark core
# ----------------------------

@dataclass
class Metrics:
    ocr_ms: List[float]
    phi_ms: List[float]
    donut_ms: List[float]
    phi_called: int = 0
    donut_called: int = 0


def run_mode(samples: List[Dict[str, Any]], mode: str, conf_threshold: float, save_csv: bool, csv_name: Optional[str]) -> None:
    results: List[Dict[str, Any]] = []
    metrics = Metrics(ocr_ms=[], phi_ms=[], donut_ms=[])

    csv_path: Optional[Path] = None
    csv_writer: Optional[csv.DictWriter] = None
    csv_fh = None

    if save_csv:
        out_dir = PROJECT_ROOT / "outputs" / "Extraction_accuracy"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = Path(csv_name) if csv_name else (out_dir / f"sroie_{mode}.csv")
        if not csv_path.is_absolute():
            csv_path = (PROJECT_ROOT / csv_path).resolve()
        csv_fh = open(csv_path, "w", newline="", encoding="utf-8")

    def write_row(row: Dict[str, Any]) -> None:
        nonlocal csv_writer
        if not save_csv or csv_fh is None:
            return
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_fh, fieldnames=list(row.keys()))
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_fh.flush()

    try:
        for i, gt in enumerate(samples, 1):
            stem = gt["stem"]
            img_path = resolve_image(stem)
            if not img_path.exists():
                continue

            print(f"🔍 {i}/{len(samples)}: Processing {stem}...", end=" ", flush=True)

            pred_vendor = None
            pred_date = None
            pred_total = None
            source = None

            latency_ocr = None
            latency_phi = None
            latency_donut = None

            raw_text = ""

            # ---- donut_only: no OCR at all ----
            if mode == "donut_only":
                t0 = time.perf_counter()
                donut = extract_fields_donut(image_path=str(img_path), task="sroie") or {}
                latency_donut = (time.perf_counter() - t0) * 1000.0
                metrics.donut_ms.append(latency_donut)
                metrics.donut_called += 1

                pred_vendor = donut.get("vendor")
                pred_date = donut.get("date")
                pred_total = donut.get("total")
                source = "donut"

            else:
                # ---- OCR always for ocr_phi / real_pipeline ----
                t0 = time.perf_counter()
                ocr = extract_fields_from_image(str(img_path)) or {}
                latency_ocr = (time.perf_counter() - t0) * 1000.0
                metrics.ocr_ms.append(latency_ocr)

                pred_vendor = ocr.get("vendor")
                pred_date = ocr.get("date")
                pred_total = ocr.get("total")
                raw_text = ocr.get("raw_text", "") or ""
                source = "ocr"

                missing = [k for k, v in [("vendor", pred_vendor), ("date", pred_date), ("total", pred_total)] if not v]

                # ---- OCR + Phi: only call Phi when OCR misses critical fields ----
                if mode in ("ocr_phi", "real_pipeline") and missing:
                    phi_out, latency_phi = parse_with_phi(raw_text)
                    metrics.phi_ms.append(latency_phi)
                    metrics.phi_called += 1

                    # fill only missing
                    if not pred_vendor and phi_out.get("vendor"):
                        pred_vendor = phi_out["vendor"]
                    if not pred_date and phi_out.get("date"):
                        pred_date = phi_out["date"]
                    if not pred_total and phi_out.get("total") is not None:
                        pred_total = phi_out["total"]
                    source = "ocr+phi"

                    # store confidences for fallback decision
                    conf_vendor = phi_out.get("conf_vendor")
                    conf_date = phi_out.get("conf_date")
                    conf_total = phi_out.get("conf_total")
                else:
                    conf_vendor = conf_date = conf_total = None

                # ---- Donut fallback (real pipeline only) ----
                if mode == "real_pipeline":
                    still_missing = [k for k, v in [("vendor", pred_vendor), ("date", pred_date), ("total", pred_total)] if not v]

                    low_conf = []
                    # Only treat confidence as low if it is present (not None)
                    if conf_vendor is not None and conf_vendor < conf_threshold:
                        low_conf.append("vendor")
                    if conf_date is not None and conf_date < conf_threshold:
                        low_conf.append("date")
                    if conf_total is not None and conf_total < conf_threshold:
                        low_conf.append("total")

                    if still_missing or low_conf:
                        t0 = time.perf_counter()
                        donut = extract_fields_donut(image_path=str(img_path), task="sroie") or {}
                        latency_donut = (time.perf_counter() - t0) * 1000.0
                        metrics.donut_ms.append(latency_donut)
                        metrics.donut_called += 1

                        # fill missing only
                        if not pred_vendor and donut.get("vendor"):
                            pred_vendor = donut["vendor"]
                        if not pred_date and donut.get("date"):
                            pred_date = donut["date"]

                        # Total: prefer Donut when it provides a plausible total and Phi/OCR disagrees.
                        d_tot = donut.get("total")
                        if d_tot is not None:
                            if pred_total is None:
                                pred_total = d_tot
                            else:
                                p_amt = _parse_amount(pred_total)
                                d_amt = _parse_amount(d_tot)
                                # If both parse and differ materially, prefer Donut (layout-aware model)
                                if p_amt is not None and d_amt is not None and abs(p_amt - d_amt) > 0.05:
                                    pred_total = d_tot

                        source = (source or "ocr") + "+donut"

            row = {
                "file": stem,
                "gt_vendor": gt.get("vendor"),
                "gt_date": gt.get("date"),
                "gt_total": gt.get("total"),
                "pred_vendor": pred_vendor,
                "pred_date": pred_date,
                "pred_total": pred_total,
                "source": source,
                "latency_ocr_ms": round(latency_ocr, 2) if latency_ocr is not None else None,
                "latency_phi_ms": round(latency_phi, 2) if latency_phi is not None else None,
                "latency_donut_ms": round(latency_donut, 2) if latency_donut is not None else None,
            }

            row["vendor_match"] = vendor_match(row.get("pred_vendor"), gt.get("vendor"))
            row["date_match"] = date_match(row.get("pred_date"), gt.get("date"))
            row["total_match"] = total_match(row.get("pred_total"), gt.get("total"))
            row["all3_match"] = bool(row["vendor_match"] and row["date_match"] and row["total_match"])

            results.append(row)
            write_row(row)

            print("✅")

    finally:
        if csv_fh is not None:
            csv_fh.close()

    # ---- summary ----
    n = len(results)
    print("\n" + "=" * 80)
    print("📈 Results Summary")
    print("=" * 80)
    print(f"N evaluated: {n}\n")

    if n == 0:
        print("No samples evaluated.")
        return

    vendor_acc = sum(1 for r in results if r["vendor_match"]) / n
    date_acc = sum(1 for r in results if r["date_match"]) / n
    total_acc = sum(1 for r in results if r["total_match"]) / n
    all3_acc = sum(1 for r in results if r["all3_match"]) / n

    print("Accuracy:")
    print(f"  Vendor: {vendor_acc*100:6.2f}%")
    print(f"  Date:   {date_acc*100:6.2f}%")
    print(f"  Total:  {total_acc*100:6.2f}%")
    print(f"  All-3:  {all3_acc*100:6.2f}%\n")

    cov_vendor = sum(1 for r in results if r.get("pred_vendor")) / n
    cov_date = sum(1 for r in results if r.get("pred_date")) / n
    cov_total = sum(1 for r in results if r.get("pred_total") is not None) / n

    print("Coverage (non-null predictions):")
    print(f"  Vendor filled: {cov_vendor*100:6.2f}%")
    print(f"  Date filled:   {cov_date*100:6.2f}%")
    print(f"  Total filled:  {cov_total*100:6.2f}%\n")

    def avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    print("Latency (avg ms):")
    if metrics.ocr_ms:
        print(f"  OCR:   {avg(metrics.ocr_ms):8.2f} ms")
    if metrics.phi_ms:
        print(f"  Phi:   {avg(metrics.phi_ms):8.2f} ms  (called on {metrics.phi_called}/{len(samples)} images)")
    if metrics.donut_ms:
        print(f"  Donut: {avg(metrics.donut_ms):8.2f} ms  (called on {metrics.donut_called}/{len(samples)} images)")
    print("")

    if save_csv and csv_path is not None:
        print(f"💾 Saved CSV to: {csv_path}\n")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SROIE benchmark: OCR vs Phi vs Donut (optimized runtime).")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", choices=["ocr_phi", "donut_only", "real_pipeline", "both"], default="both")
    parser.add_argument("--conf_threshold", type=float, default=0.60, help="Trigger Donut if Phi confidence below this (only if confidence is present).")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--csv_name", type=str, default=None)
    args = parser.parse_args()

    print(f"📊 Loading SROIE-100 samples (limit={args.limit})...")
    samples = load_sroie_samples(args.limit)
    print(f"   Loaded {len(samples)} samples\n")

    if args.mode == "both":
        # run twice, printing two summaries; CSV (if enabled) will use separate files
        run_mode(samples, "ocr_phi", args.conf_threshold, args.save_csv, args.csv_name or f"outputs/Extraction_accuracy/sroie_ocr_phi.csv")
        print("\n\n")
        run_mode(samples, "real_pipeline", args.conf_threshold, args.save_csv, args.csv_name or f"outputs/Extraction_accuracy/sroie_real_pipeline.csv")
        return

    run_mode(samples, args.mode, args.conf_threshold, args.save_csv, args.csv_name)


if __name__ == "__main__":
    main()
