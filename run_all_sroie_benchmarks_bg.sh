#!/usr/bin/env bash
set -e

mkdir -p outputs/logs
STAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting SROIE benchmarks in background (stamp=$STAMP)"
echo "Logs: outputs/logs/"
echo ""

nohup python scripts/eval_sroie_extraction_compare.py \
  --limit 100 \
  --mode ocr_phi \
  --save_csv \
  --csv_name "outputs/sroie_compare_results_ocr_phi_${STAMP}.csv" \
  > "outputs/logs/sroie_ocr_phi_${STAMP}.log" 2>&1 &

nohup python scripts/eval_sroie_extraction_compare.py \
  --limit 100 \
  --mode donut_only \
  --save_csv \
  --csv_name "outputs/sroie_compare_results_donut_only_${STAMP}.csv" \
  > "outputs/logs/sroie_donut_only_${STAMP}.log" 2>&1 &

nohup python scripts/eval_sroie_extraction_compare.py \
  --limit 100 \
  --mode real_pipeline \
  --save_csv \
  --csv_name "outputs/sroie_compare_results_real_pipeline_${STAMP}.csv" \
  > "outputs/logs/sroie_real_pipeline_${STAMP}.log" 2>&1 &

echo "Launched background jobs:"
jobs -l || true

echo ""
echo "Monitor progress:"
echo "  tail -f outputs/logs/sroie_ocr_phi_${STAMP}.log"
echo "  tail -f outputs/logs/sroie_donut_only_${STAMP}.log"
echo "  tail -f outputs/logs/sroie_real_pipeline_${STAMP}.log"

echo ""
echo "CSV outputs:"
echo "  outputs/sroie_compare_results_ocr_phi_${STAMP}.csv"
echo "  outputs/sroie_compare_results_donut_only_${STAMP}.csv"
echo "  outputs/sroie_compare_results_real_pipeline_${STAMP}.csv"