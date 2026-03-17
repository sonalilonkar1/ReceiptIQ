"""Quick smoke test for ReceiptIQ pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.db import find_duplicates, insert_document, spend_by_vendor
from scripts.init_db import init_db


def run_smoke_test() -> None:
    """Run a minimal end-to-end DB pipeline check."""
    print("== ReceiptIQ Smoke Test ==")

    init_db()

    fake_extracted = {
        "doc_type": "receipt",
        "vendor": "Acme Market",
        "doc_date": "03/16/2026",
        "currency": "USD",
        "subtotal": 18.50,
        "tax": 1.50,
        "total": 20.00,
        "confidence": 0.9,
        "raw_text": "ACME MARKET\n03/16/2026\nSUBTOTAL 18.50\nTAX 1.50\nTOTAL 20.00",
    }

    doc_id_1 = insert_document(fake_extracted)
    doc_id_2 = insert_document(fake_extracted)
    print(f"Inserted doc IDs: {doc_id_1}, {doc_id_2}")

    spend_rows = spend_by_vendor()
    print("\nSpend by vendor:")
    for row in spend_rows:
        print(row)

    duplicate_rows = find_duplicates()
    print("\nDuplicate candidates:")
    for row in duplicate_rows:
        print(row)


if __name__ == "__main__":
    run_smoke_test()
