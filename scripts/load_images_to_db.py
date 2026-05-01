from pathlib import Path
from app.tools.vision import extract_fields_from_image, validate_totals
from app.tools.db import insert_document, add_flag

IMG_DIR = Path("data/cord_100_clean/images")  # use clean set
# IMG_DIR = Path("data/cord_100/images")      # or use raw set

def to_payload(extracted: dict) -> dict:
    # matches your agent.py mapping
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

def main():
    imgs = sorted(list(IMG_DIR.glob("*.png")))
    if not imgs:
        raise SystemExit(f"No images found in {IMG_DIR}. Run download + filter first.")

    ok = 0
    flagged = 0

    for p in imgs:
        extracted = extract_fields_from_image(str(p))
        doc_id = insert_document(to_payload(extracted))

        mismatch = validate_totals(extracted)
        if mismatch:
            add_flag(doc_id, "totals_validation", mismatch)
            flagged += 1
        else:
            ok += 1

    print(f"Ingested {len(imgs)} docs. Valid={ok}, Flagged={flagged}")

if __name__ == "__main__":
    main()