# scripts/load_sroie_to_db.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.vision import extract_fields_from_image, validate_totals
from app.tools.db import insert_document, add_flag


def to_payload(extracted: dict) -> dict:
    """
    Match the payload shape your agent uses in _to_document_payload().
    """
    return {
        "doc_type": extracted.get("doc_type", "receipt"),
        "vendor": extracted.get("vendor"),
        "doc_date": extracted.get("date"),
        "currency": extracted.get("currency", "USD"),
        "subtotal": extracted.get("subtotal"),
        "tax": extracted.get("tax"),
        "total": extracted.get("total"),
        "confidence": extracted.get("confidence_overall"),
        "category": extracted.get("category", "other"),
        "line_items": extracted.get("line_items"),
        "description": extracted.get("description"),
        "invoice_number": extracted.get("invoice_number"),
        "raw_text": extracted.get("raw_text"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SROIE images into ReceiptIQ SQLite DB.")
    parser.add_argument("--img_dir", type=str, default="data/sroie_100/images", help="Directory containing SROIE images")
    parser.add_argument("--limit", type=int, default=100, help="Max number of images to ingest")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    if not imgs:
        raise SystemExit(f"No images found in {img_dir}. Run download_sroie_100.py first.")

    imgs = imgs[: args.limit]

    ingested = 0
    flagged = 0

    for p in imgs:
        extracted = extract_fields_from_image(str(p))
        doc_id = insert_document(to_payload(extracted))

        mismatch = validate_totals(extracted)
        if mismatch:
            add_flag(doc_id, "totals_validation", mismatch)
            flagged += 1

        ingested += 1
        if ingested % 25 == 0:
            print(f"Ingested {ingested}/{len(imgs)}")

    print(f"\nDone. Ingested {ingested} docs. Flagged={flagged}.")


if __name__ == "__main__":
    main()