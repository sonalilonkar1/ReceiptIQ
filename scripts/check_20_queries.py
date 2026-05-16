# scripts/check_20_queries_patched.py
"""
ReceiptIQ - Planned 20 Query Coverage Check (Patched)

What this script does:
- Runs your 20 planned queries in a way that matches the *real app*:
  - Q1 is tested as a FILE UPLOAD ingest (not a text query).
- Checks each query against an EXPECTED INTENT.
- Flags mismatches clearly.
- Also checks: citations present when tools should be used.

Run:
  python scripts/check_20_queries_patched.py
Optional:
  python scripts/check_20_queries_patched.py --img data/sroie_100/images/sroie_train_00000.jpg
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple


# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app.agent import handle_message

def has_citations(r: Dict[str, Any]) -> bool:
    c = r.get("citations")
    return isinstance(c, list) and len(c) > 0


def tool_signals(dbg: Dict[str, Any]) -> Dict[str, bool]:
    # Signals used across your repo for DB/tool grounding
    return {
        "rows": dbg.get("rows") is not None,
        "duplicates": dbg.get("duplicates") is not None,
        "anomalies": dbg.get("anomalies") is not None,
        "categories": dbg.get("categories") is not None,
        "weekly": dbg.get("weekly") is not None or dbg.get("weekly_totals") is not None,
        "monthly": dbg.get("monthly") is not None or dbg.get("monthly_totals") is not None,
    }


# Each test: (id, type, query/file, expected_intent, needs_citations)
# needs_citations=True means: should be grounded in DB/WEB and cite.
TESTS: List[Dict[str, Any]] = [
    # 01 - Must be upload flow in the real app
    {
        "id": 1,
        "kind": "upload",
        "prompt": "",  # message can be empty for upload
        "expected_intent": "file_ingest",
        "needs_citations": True,
        "desc": "Upload receipt → extract vendor/date/total → DB insert",
    },
    {
        "id": 2,
        "kind": "text",
        "prompt": "Is the total consistent with subtotal + tax? Flag any mismatch.",
        "expected_intent": "validate_totals",
        "needs_citations": True,
        "desc": "Validate totals (subtotal+tax vs total) + flag",
    },
    {
        "id": 3,
        "kind": "text",
        "prompt": "List all receipts saved this week and their totals.",
        "expected_intent": "weekly_summary",
        "needs_citations": True,
        "desc": "Weekly list / grouping from DB",
    },
    {
        "id": 4,
        "kind": "text",
        "prompt": "How much did I spend last month? Break it down by vendor.",
        "expected_intent": "spend_by_vendor",
        "needs_citations": True,
        "desc": "Spend by vendor, last month",
    },
    {
        "id": 5,
        "kind": "text",
        "prompt": "Show my top 5 spending categories this month.",
        "expected_intent": "spending_by_category",
        "needs_citations": True,
        "desc": "Top categories (limit=5), this month",
    },
    {
        "id": 6,
        "kind": "text",
        "prompt": "Find duplicate receipts (same vendor/date/total) and flag them.",
        "expected_intent": "duplicates",
        "needs_citations": True,
        "desc": "Duplicate detection",
    },
    {
        "id": 7,
        "kind": "text",
        "prompt": "Which documents have missing date or vendor fields?",
        "expected_intent": "missing_fields",
        "needs_citations": True,
        "desc": "Missing fields list",
    },
    {
        "id": 8,
        "kind": "text",
        "prompt": "Show all receipts over $100 in the last 90 days.",
        "expected_intent": "threshold_search",
        "needs_citations": True,
        "desc": "Threshold search + date window",
    },
    {
        "id": 9,
        "kind": "text",
        "prompt": "Find all receipts containing 'parking' in line items.",
        "expected_intent": "keyword_search",
        "needs_citations": True,
        "desc": "Keyword search in raw_text/line_items",
    },
    {
        "id": 10,
        "kind": "text",
        "prompt": "What’s my average lunch spend per week?",
        "expected_intent": "avg_lunch_weekly",
        "needs_citations": True,
        "desc": "Average lunch per week (category filter + avg)",
    },
    {
        "id": 11,
        "kind": "text",
        "prompt": "Create a reimbursement summary for Feb 1–Feb 15 with totals by category.",
        "expected_intent": "reimbursement_summary",
        "needs_citations": True,
        "desc": "Reimbursement summary (date range + category totals)",
    },
    {
        "id": 12,
        "kind": "text",
        "prompt": "Draft an email to my manager summarizing reimbursable expenses for this period.",
        "expected_intent": "reimbursement_email",
        "needs_citations": True,
        "desc": "Email draft grounded in DB totals",
    },
    {
        "id": 13,
        "kind": "text",
        "prompt": "Mark these receipts as ‘reimbursable’ and generate a checklist of required attachments.",
        "expected_intent": "mark_reimbursable",
        "needs_citations": True,
        "desc": "DB update + checklist output",
    },
    {
        "id": 14,
        "kind": "text",
        "prompt": "This receipt is in EUR—convert it to USD using the rate on the purchase date.",
        "expected_intent": "web_lookup",
        "needs_citations": True,
        "desc": "Currency conversion via web tool",
    },
    {
        "id": 15,
        "kind": "text",
        "prompt": "Verify this vendor’s official website/contact info and store it.",
        "expected_intent": "vendor_verification",
        "needs_citations": True,
        "desc": "Vendor verification via web tool (and store result)",
    },
    {
        "id": 16,
        "kind": "text",
        "prompt": "Which receipts violate a $25 lunch limit policy?",
        "expected_intent": "rule_violations",
        "needs_citations": True,
        "desc": "Policy rule violations (DB)",
    },
    {
        "id": 17,
        "kind": "text",
        "prompt": "Flag suspicious invoices: missing invoice number, missing address, or unusual totals.",
        "expected_intent": "anomalies",
        "needs_citations": True,
        "desc": "Anomaly detection",
    },
    {
        "id": 18,
        "kind": "text",
        "prompt": "Compare spending between January and February by category.",
        "expected_intent": "compare_spending_periods",
        "needs_citations": True,
        "desc": "Compare two periods by category",
    },
    {
        "id": 19,
        "kind": "text",
        "prompt": "Generate a CSV-style summary of all receipts for my expense report.",
        "expected_intent": "export_csv",
        "needs_citations": True,
        "desc": "Export summary",
    },
    {
        "id": 20,
        "kind": "text",
        "prompt": "Explain why a receipt was flagged and what I should do next.",
        "expected_intent": "explain_flag",
        "needs_citations": True,
        "desc": "Explain audit flag + next steps",
    },
]


def run_one(test: Dict[str, Any], upload_img: str) -> Tuple[bool, Dict[str, Any]]:
    """Return (pass, details)."""
    kind = test["kind"]
    expected_intent = test["expected_intent"]

    if kind == "upload":
        r = handle_message(test["prompt"], file_path=upload_img)
    else:
        r = handle_message(test["prompt"])

    dbg = r.get("debug", {}) if isinstance(r, dict) else {}
    intent = (dbg.get("intent") or "").strip()

    citations_ok = (not test["needs_citations"]) or has_citations(r)
    intent_ok = (intent == expected_intent)

    # Collect tool grounding signals (helpful for diagnosing false positives)
    sig = tool_signals(dbg)

    passed = intent_ok and citations_ok

    details = {
        "id": test["id"],
        "desc": test["desc"],
        "expected_intent": expected_intent,
        "actual_intent": intent,
        "citations": r.get("citations"),
        "citations_ok": citations_ok,
        "intent_ok": intent_ok,
        "tool_signals": sig,
        "latency_ms": dbg.get("latency_ms") or r.get("latency_ms"),
    }
    return passed, details


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="data/sroie_100/images/sroie_train_00000.jpg",
                    help="Image path to use for Q1 upload ingest test.")
    args = ap.parse_args()

    print("\n=== ReceiptIQ 20-Query Coverage Check (Patched) ===")
    print(f"Upload test image for Q1: {args.img}\n")

    total = 0
    ok = 0
    mismatches: List[Dict[str, Any]] = []

    for t in TESTS:
        total += 1
        passed, d = run_one(t, upload_img=args.img)

        status = "✅" if passed else "❌"
        print(f"[{d['id']:02d}] {status} {d['desc']}")
        print(f"     expected_intent={d['expected_intent']}  actual_intent={d['actual_intent']}")
        print(f"     citations_ok={d['citations_ok']}  citations={d['citations']}")
        print(f"     tool_signals={d['tool_signals']}  latency_ms={d['latency_ms']}\n")

        if passed:
            ok += 1
        else:
            mismatches.append(d)

    print("=== Summary ===")
    print(f"Passed: {ok}/{total}")
    if mismatches:
        print("\nMismatches to fix:")
        for d in mismatches:
            print(f"  - Q{d['id']:02d}: expected {d['expected_intent']} but got {d['actual_intent']} (citations_ok={d['citations_ok']})")

    # Exit code helpful for CI
    raise SystemExit(0 if ok == total else 1)


if __name__ == "__main__":
    main()