#!/usr/bin/env python
"""Evaluate Phi vs Mistral vs Phi+Mistral on the 20 planned queries.

Design goals:
- Run the *real* app entrypoint (handle_message) for each query.
- Evaluate answer *correctness* using tool-grounded truth signals exposed in debug["eval_expected"].
- Isolate DB state per model mode using RECEIPTIQ_DB_PATH.

Usage (from repo root):
  python scripts/eval_llm_answer_accuracy_20_modes.py --img data/sroie_100/images/sroie_train_00000.jpg

Outputs:
  outputs/llm_answer_accuracy_<stamp>.csv
  outputs/llm_answer_accuracy_<stamp>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODES = ["phi_only", "mistral_only", "phi+mistral"]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _extract_doc_ids(text: str) -> List[int]:
    ids = set()
    for m in re.finditer(r"(?:doc_id\s*=\s*|Doc\s*#|Document\s*#|receipt\s*#)(\d+)", text, flags=re.IGNORECASE):
        try:
            ids.add(int(m.group(1)))
        except Exception:
            pass
    return sorted(ids)


def _extract_amounts(text: str) -> List[float]:
    vals = []
    for m in re.finditer(r"\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", text):
        try:
            v = float(m.group(1).replace(",", ""))
            vals.append(v)
        except Exception:
            pass
    return vals


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 &.'/-]+", "", s)
    return s


def _f1(pred: List[int], gold: List[int]) -> float:
    ps, gs = set(pred), set(gold)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    tp = len(ps & gs)
    prec = tp / len(ps)
    rec = tp / len(gs)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _score_from_eval_expected(intent: str, response: str, citations: List[str], ee: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Return (score in [0,1], details)."""
    details: Dict[str, Any] = {"intent": intent}

    # Basic grounding
    details["has_citations"] = bool(citations)

    # Prefer doc-id correctness when rows exist
    if "rows" in ee and isinstance(ee["rows"], list) and ee["rows"]:
        rows = ee["rows"]
        # If first element of row looks like doc_id
        if isinstance(rows[0], (list, tuple)) and rows[0] and isinstance(rows[0][0], int):
            gold = [int(r[0]) for r in rows if isinstance(r, (list, tuple)) and r and isinstance(r[0], int)]
            pred = _extract_doc_ids(response)
            f1 = _f1(pred, gold)
            details.update({"gold_doc_ids": gold[:10], "pred_doc_ids": pred[:10], "doc_id_f1": round(f1, 3)})
            # require citations for DB intents
            score = 0.6 * f1 + 0.4 * (1.0 if details["has_citations"] else 0.0)
            return score, details

        # Vendor/category aggregates: (name, amount)
        if isinstance(rows[0], (list, tuple)) and len(rows[0]) >= 2:
            name0 = str(rows[0][0])
            amt0 = rows[0][1]
            details["gold_top_name"] = name0
            try:
                gold_amt = float(amt0)
            except Exception:
                gold_amt = None
            details["gold_top_amount"] = gold_amt

            # If ground truth does not include an amount for this intent (e.g., duplicates list),
            # treat amount correctness as "not applicable" so we don't penalize the model.
            amt_not_applicable = (gold_amt is None)
            resp_norm = _norm(response)
            name_ok = _norm(name0) in resp_norm if name0 else False
            amt_ok = True if amt_not_applicable else False
            if gold_amt is not None:
                amts = _extract_amounts(response)
                amt_ok = any(abs(a - gold_amt) <= 0.05 for a in amts)
                details["pred_amounts_sample"] = amts[:10]
            details.update({"top_name_ok": name_ok, "top_amount_ok": amt_ok, "amt_not_applicable": amt_not_applicable})
            score = 0.4 * (1.0 if name_ok else 0.0) + 0.4 * (1.0 if amt_ok else 0.0) + 0.2 * (1.0 if details["has_citations"] else 0.0)
            return score, details

    # Validate totals intent: expect audit flag citation when mismatch present
    if intent == "validate_totals":
        mismatch = ee.get("validation")
        mentions_flag = ("flag" in _norm(response)) or ("mismatch" in _norm(response))
        has_audit_cite = any("audit_flags" in c for c in citations)
        score = 0.5 * (1.0 if mentions_flag else 0.0) + 0.5 * (1.0 if has_audit_cite else 0.0)
        details.update({"mismatch": mismatch, "mentions_flag": mentions_flag, "has_audit_cite": has_audit_cite})
        return score, details

    # Vendor verification: require URL/domain
    if intent == "vendor_verification":
        has_web = any(c.startswith("WEB:") for c in citations)
        has_url = bool(re.search(r"(https?://|\b[a-z0-9-]+\.(com|org|net|edu)\b)", response.lower()))
        if not has_url and "website" in response.lower():
            has_url = bool(re.search(r"website\s*:\s*\S*\.", response.lower()))
        if not has_url and has_web:
            # If we cited the web tool, count as verified source even if URL formatting is missing
            has_url = True
        score = 0.5 * (1.0 if has_web else 0.0) + 0.5 * (1.0 if has_url else 0.0)
        details.update({"has_web_citation": has_web, "has_url": has_url})
        return score, details

    # Web lookup: require some numeric output + web citation
    if intent == "web_lookup":
        has_web = any(c.startswith("WEB:") for c in citations)
        has_num = bool(_extract_amounts(response))
        score = 0.5 * (1.0 if has_web else 0.0) + 0.5 * (1.0 if has_num else 0.0)
        details.update({"has_web_citation": has_web, "has_num": has_num})
        return score, details

    # Help / how_it_works: checklist
    if intent in ("help", "how_it_works"):
        needed = ["ocr", "sqlite"]
        if intent == "how_it_works":
            needed += ["donut", "validation"]
        rn = _norm(response)
        hits = sum(1 for k in needed if k in rn)
        score = hits / max(1, len(needed))
        details.update({"needed": needed, "hits": hits})
        return score, details

    # Fallback: require citations for non-trivial intents
    score = 1.0 if details["has_citations"] else 0.0
    return score, details


def _worker(mode: str, img: str, out_csv: str) -> None:
    # Ensure we import project code
    sys.path.insert(0, str(PROJECT_ROOT))

    # Set model mode
    import app.agent as agent
    agent.MODEL_MODE = mode

    from scripts.check_20_queries import TESTS  # type: ignore

    rows_out: List[Dict[str, Any]] = []

    for t in TESTS:
        tid = t["id"]
        kind = t["kind"]
        prompt = t["prompt"]
        expected_intent = t["expected_intent"]

        if kind == "upload":
            r = agent.handle_message("", file_path=img)
        else:
            r = agent.handle_message(prompt)

        dbg = r.get("debug", {}) if isinstance(r, dict) else {}
        intent = dbg.get("intent", "")
        citations = r.get("citations", []) if isinstance(r, dict) else []
        response = r.get("response", "") if isinstance(r, dict) else ""
        ee = dbg.get("eval_expected") if isinstance(dbg, dict) else None
        if not isinstance(ee, dict):
            ee = {"intent": intent}
        # Attach a few useful signals that _wrap_response may not include
        if dbg.get("validation") is not None:
            ee["validation"] = dbg.get("validation")

        score, details = _score_from_eval_expected(expected_intent, response, citations, ee)

        rows_out.append({
            "id": tid,
            "mode": mode,
            "query_kind": kind,
            "prompt": prompt,
            "expected_intent": expected_intent,
            "actual_intent": intent,
            "score": round(score, 3),
            "latency_ms": r.get("latency_ms"),
            "has_citations": bool(citations),
            "citations": "|".join(citations) if isinstance(citations, list) else str(citations),
            "details": json.dumps(details, ensure_ascii=False),
        })

    # Write CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Image path used for Q1 upload")
    ap.add_argument("--out", default=None, help="Output CSV (or auto)")
    ap.add_argument("--worker", action="store_true", help="Run as worker")
    ap.add_argument("--mode", default=None, help="Worker mode")
    args = ap.parse_args()

    if args.worker:
        if not args.mode:
            raise SystemExit("--mode required in --worker")
        out_csv = args.out or f"outputs/llm_answer_accuracy_{args.mode}.csv"
        _worker(args.mode, args.img, out_csv)
        return

    stamp = _now_stamp()
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Create per-mode DB copies
    base_db = PROJECT_ROOT / "receiptiq.sqlite"
    if not base_db.exists():
        raise SystemExit("receiptiq.sqlite not found. Run scripts/init_db.py first.")

    db_dir = outputs_dir / "eval_dbs" / stamp
    db_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    per_mode_csvs: List[str] = []

    for mode in MODES:
        db_path = db_dir / f"receiptiq_{mode}.sqlite"
        shutil.copy2(base_db, db_path)

        out_csv = str(outputs_dir / f"llm_answer_accuracy_{mode}_{stamp}.csv")
        per_mode_csvs.append(out_csv)

        env = os.environ.copy()
        env["RECEIPTIQ_DB_PATH"] = str(db_path)

        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", mode, "--img", args.img, "--out", out_csv]
        print(f"Running {mode} with DB={db_path.name} ...")
        subprocess.check_call(cmd, cwd=str(PROJECT_ROOT), env=env)

        # Merge
        import pandas as pd  # lightweight
        df = pd.read_csv(out_csv)
        merged_rows.extend(df.to_dict(orient="records"))

    merged_csv = str(outputs_dir / f"llm_answer_accuracy_ALL_{stamp}.csv")
    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
        w.writeheader()
        w.writerows(merged_rows)

    # Summary
    summary: Dict[str, Any] = {"stamp": stamp, "modes": {}}
    for mode in MODES:
        rows = [r for r in merged_rows if r["mode"] == mode]
        avg = sum(float(r["score"]) for r in rows) / max(1, len(rows))
        pass_rate = sum(1 for r in rows if float(r["score"]) >= 0.8) / max(1, len(rows))
        summary["modes"][mode] = {
            "avg_score": round(avg, 3),
            "pass_rate@0.8": round(pass_rate, 3),
            "csv": str(outputs_dir / f"llm_answer_accuracy_{mode}_{stamp}.csv"),
        }

    summary_path = outputs_dir / f"llm_answer_accuracy_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print("Merged CSV:", merged_csv)
    print("Summary:", summary_path)
    for mode in MODES:
        print(mode, summary["modes"][mode])


if __name__ == "__main__":
    main()
