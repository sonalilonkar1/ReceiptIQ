#!/usr/bin/env bash
set -euo pipefail

# Default to 50 images for all modes (you can override: ./run_extraction_accuracy_summary.sh 30)
LIMIT="${1:-50}"

OUTDIR="outputs/Extraction_accuracy"
LOGDIR="${OUTDIR}/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

STAMP="$(date +"%Y%m%d_%H%M%S")"
SUMMARY="${OUTDIR}/summary_${STAMP}.csv"

echo "Running extraction accuracy SUMMARY suite (limit=${LIMIT})"
echo "Logs: ${LOGDIR}"
echo "Summary CSV: ${SUMMARY}"
echo ""

# CSV header (now includes avg_phi_ms)
echo "mode,limit,n_evaluated,vendor_acc,date_acc,total_acc,all3_acc,vendor_filled,date_filled,total_filled,avg_ocr_ms,avg_phi_ms,avg_donut_ms,donut_called" > "${SUMMARY}"

append_summary () {
  MODE="$1"
  LOG="$2"

  python - <<'PY' "$MODE" "$LOG" "$LIMIT" >> "${SUMMARY}"
import re, sys, pathlib

mode = sys.argv[1]
log_path = pathlib.Path(sys.argv[2])
limit = sys.argv[3]
text = log_path.read_text(errors="ignore")

def g(pat, default=""):
    m = re.search(pat, text, re.MULTILINE)
    return m.group(1).strip() if m else default

n = g(r"N evaluated:\s*([0-9]+)", "0")

# Compare-script style
vendor_acc = g(r"Vendor:\s*([0-9.]+%)", "")
date_acc   = g(r"Date:\s*([0-9.]+%)", "")
total_acc  = g(r"Total:\s*([0-9.]+%)", "")
all3_acc   = g(r"All-3:\s*([0-9.]+%)", "")

vendor_fill = g(r"Vendor filled:\s*([0-9.]+%)", "")
date_fill   = g(r"Date filled:\s*([0-9.]+%)", "")
total_fill  = g(r"Total filled:\s*([0-9.]+%)", "")

avg_ocr   = g(r"OCR:\s*([0-9.]+)\s*ms", "")
avg_phi   = g(r"Phi:\s*([0-9.]+)\s*ms", "")
avg_donut = g(r"Donut:\s*([0-9.]+)\s*ms", "")
donut_called = g(r"called on\s*([0-9]+/[0-9]+)\s*images", "")

# Fallback for eval_sroie_extraction.py style (if you ever include it)
if not vendor_acc:
    vendor_acc = g(r"Vendor.*?:\s*([0-9.]+%)", "")
if not date_acc:
    date_acc = g(r"Date.*?:\s*([0-9.]+%)", "")
if not total_acc:
    total_acc = g(r"Total.*?:\s*([0-9.]+%)", "")
if not all3_acc:
    all3_acc = g(r"All three.*?:\s*([0-9.]+%)", "")

print(f"{mode},{limit},{n},{vendor_acc},{date_acc},{total_acc},{all3_acc},{vendor_fill},{date_fill},{total_fill},{avg_ocr},{avg_phi},{avg_donut},{donut_called}")
PY
}

run_step () {
  MODE="$1"
  CMD="$2"
  LOG="${LOGDIR}/${MODE}_${STAMP}.log"

  echo "============================================================"
  echo "STEP: ${MODE}"
  echo "CMD : ${CMD}"
  echo "LOG : ${LOG}"
  echo "============================================================"

  bash -lc "${CMD}" 2>&1 | tee "${LOG}"
  echo ""

  append_summary "${MODE}" "${LOG}"
  echo "✅ Added ${MODE} to summary CSV"
  echo ""
}

# Runs your current compare script modes (supported: ocr_phi, donut_only, real_pipeline, both)
run_step "01_ocr_phi"        "python scripts/eval_sroie_extraction_compare.py --limit ${LIMIT} --mode ocr_phi"
run_step "02_donut_only"     "python scripts/eval_sroie_extraction_compare.py --limit ${LIMIT} --mode donut_only"
run_step "03_real_pipeline"  "python scripts/eval_sroie_extraction_compare.py --limit ${LIMIT} --mode real_pipeline"
run_step "04_both"           "python scripts/eval_sroie_extraction_compare.py --limit ${LIMIT} --mode both"

echo "============================================================"
echo "DONE ✅"
echo "Summary: ${SUMMARY}"
echo ""
echo "Preview:"
cat "${SUMMARY}"
echo "============================================================"