import sys
from pathlib import Path
import shutil
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.tools.vision import extract_fields_from_image
from app.storage import save_receipt

SRC = Path("data/cord_100/images")
DST = Path("data/cord_100_clean/images")
TARGET = 100  # try to keep up to 100 clean ones

def is_clean(extracted: dict) -> bool:
    # Require 2 of 3 key fields: vendor, date, or total
    has_vendor = bool(extracted.get("vendor"))
    has_date = bool(extracted.get("date"))
    has_total = extracted.get("total") is not None
    
    # Count how many fields we have
    field_count = sum([has_vendor, has_date, has_total])
    return field_count >= 2

def main():
    DST.mkdir(parents=True, exist_ok=True)
    kept = 0
    pending_review_images = []

    for img_path in sorted(SRC.glob("*.png")):
        if kept >= TARGET:
            break
        extracted = extract_fields_from_image(str(img_path))
        if is_clean(extracted):
            shutil.copy2(img_path, DST / img_path.name)
            kept += 1
            
            # Save to database
            save_receipt(img_path.name, extracted)
            
            # Track if this image has pending review items
            if extracted.get("pending_review"):
                pending_review_images.append({
                    "image": img_path.name,
                    "missing_fields": extracted.get("missing_fields"),
                    "confidence": extracted.get("confidence_overall")
                })

    print(f"Kept {kept} clean images at: {DST.resolve()}")
    
    if pending_review_images:
        print(f"\n⚠️  {len(pending_review_images)} images need manual review (missing fields):")
        for item in pending_review_images[:10]:  # Show first 10
            print(f"  - {item['image']}: missing {item['missing_fields']} (confidence: {item['confidence']})")
        if len(pending_review_images) > 10:
            print(f"  ... and {len(pending_review_images) - 10} more")

if __name__ == "__main__":
    main()