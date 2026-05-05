"""Vision-related utilities (OCR, image parsing)."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

from PIL import Image, ImageOps, ImageFilter
import pytesseract




_DATE_REGEX = re.compile(
    r"\b(?:"
    r"\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}"
    r"|\d{4}[\/-]\d{1,2}[\/-]\d{1,2}"
    r")\b"
)
_AMOUNT_REGEX = r"(-?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})|-?\$?\s*\d+(?:\.\d{2}))"


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for robust OCR: grayscale, autocontrast, upscale, threshold.
    
    Uses OpenCV threshold if available; falls back to PIL filtering otherwise.
    """
    # Convert to grayscale
    image = image.convert("L")
    
    # Apply autocontrast to improve clarity
    image = ImageOps.autocontrast(image)
    
    # Upscale 2x for better OCR accuracy
    width, height = image.size
    image = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
    
    # Try to apply threshold using OpenCV for best quality
    try:
        import cv2
        import numpy as np
        
        img_array = np.array(image)
        _, img_array = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY)
        image = Image.fromarray(img_array)
    except (ImportError, Exception):
        # Fallback: apply PIL threshold filter if cv2 unavailable
        image = image.point(lambda x: 0 if x < 150 else 255, '1').convert('L')
    
    return image


def _preprocess_image_simple(image: Image.Image) -> Image.Image:
    """Fallback preprocessing when cv2 unavailable: grayscale, autocontrast, upscale."""
    # Convert to grayscale
    image = image.convert("L")
    
    # Apply autocontrast to improve clarity
    image = ImageOps.autocontrast(image)
    
    # Upscale 2x for better OCR accuracy
    width, height = image.size
    image = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
    
    return image


def _ocr_region(image: Image.Image, top_percent: float = 0.0, bottom_percent: float = 1.0) -> str:
    """OCR a specific vertical region of the image using enhanced config."""
    # Crop region by percentage (0.0 = top, 1.0 = bottom)
    width, height = image.size
    top = int(height * top_percent)
    bottom = int(height * bottom_percent)
    
    cropped = image.crop((0, top, width, bottom))
    
    # Use enhanced Tesseract config: OEM 3 (default) + PSM 6 (uniform text blocks)
    text = pytesseract.image_to_string(cropped, config="--oem 3 --psm 6")
    return text


def _find_max_amount(raw_text: str) -> Optional[float]:
    """Extract the maximum money amount from text as fallback for total."""
    matches = re.findall(_AMOUNT_REGEX, raw_text)
    if not matches:
        return None
    
    amounts = []
    for match in matches:
        amount = _normalize_amount(match)
        if amount is not None and amount > 0:
            amounts.append(amount)
    
    return max(amounts) if amounts else None


def _normalize_amount(raw: str) -> Optional[float]:
    cleaned = raw.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_amount_by_labels(raw_text: str, labels: list[str]) -> Optional[float]:
    """Find amount by labels, ignoring 'total boxes' lines for robustness."""
    # Filter out lines containing "total boxes" (crumpled receipt artifact)
    filtered_lines = []
    for line in raw_text.splitlines():
        if "total boxes" not in line.lower():
            filtered_lines.append(line)
    
    filtered_text = "\n".join(filtered_lines)
    
    for label in labels:
        pattern = re.compile(rf"(?im)^.*\b{label}\b[^\d-]*{_AMOUNT_REGEX}.*$")
        for match in pattern.finditer(filtered_text):
            value = _normalize_amount(match.group(1))
            if value is not None:
                return value

    # Fallback: look for inline occurrences when OCR line breaks are noisy.
    for label in labels:
        pattern = re.compile(rf"(?i)\b{label}\b[^\d-]*{_AMOUNT_REGEX}")
        match = pattern.search(filtered_text)
        if match:
            value = _normalize_amount(match.group(1))
            if value is not None:
                return value

    return None


def _guess_vendor(lines: list[str]) -> Optional[str]:
    """Extract vendor name from receipt, prioritizing brand names over service types.
    
    Strategy:
    1. Skip obvious service types and annotations
    2. Prefer multi-word candidates (real brands are often 2+ words)
    3. Skip single words that look like OCR errors
    4. Skip lines with timestamps, codes, or common receipt metadata
    5. Look through more lines to find actual brand names
    """
    # Skip tokens that indicate service type, not vendor name
    skip_tokens = (
        "receipt", "invoice", "tax", "date", "total", "subtotal",
        "carry-out", "carry out", "carryout", "dine-in", "dine in", "dining",
        "takeout", "take out", "delivery", "order", "not paid", "paid",
        "server", "table", "check", "bill", "tip", "save", "email", "website"
    )
    
    # Metadata keywords to skip when they're the first word
    metadata_first_words = (
        "order", "server", "table", "ref", "id", "store", "location", "address",
        "street", "city", "state", "zip", "phone", "register", "terminal"
    )
    
    candidates = []
    
    # Look through first 20 lines
    for line in lines[:20]:
        candidate = re.sub(r"[^A-Za-z0-9&.,'\- ]+", "", line).strip()
        if len(candidate) < 3:
            continue
        lowered = candidate.lower()
        
        # Skip known service types/annotations
        if any(token in lowered for token in skip_tokens):
            continue
        
        # Skip if contains common codes/IDs (all uppercase + numbers, like "KRV4BFHZ")
        if re.match(r"^[A-Z0-9]{5,}$", candidate):
            continue
        
        # Skip pure phone/numbers
        if re.match(r"^\+?\d[\d\-\(\)\s]{5,}$", candidate):
            continue
        
        # Skip if starts with a number (likely an address like "123 Main St")
        if candidate[0].isdigit():
            continue
        
        # Skip if starts with metadata keyword
        first_word = candidate.split()[0].lower()
        if first_word in metadata_first_words:
            continue
        
        # Skip lines that are mostly numbers/single words with numbers (like "Order 1065860")
        if re.search(r"\d", candidate):
            word_count = len(candidate.split())
            if word_count <= 1:
                continue
        
        # Remove trailing store/location numbers (4-5 digits: "Domino's Pizza 3693" -> "Domino's Pizza")
        candidate_clean = re.sub(r'\s+\d{4,5}$', '', candidate).strip()
        if not candidate_clean:
            candidate_clean = candidate
        
        # Collect candidate with word count (prefer multi-word names)
        word_count = len(candidate_clean.split())
        candidates.append((word_count, len(candidate_clean), candidate_clean))
    
    # Sort by word count (prefer 2+ words), then by length (prefer longer names)
    if candidates:
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        return candidates[0][2]
    
    return None


def _extract_date(raw_text: str) -> Optional[str]:
    match = _DATE_REGEX.search(raw_text)
    return match.group(0) if match else None


def _classify_category(vendor: Optional[str], raw_text: str) -> str:
    """Classify receipt into category: meals, travel, supplies, or other."""
    if not vendor:
        vendor = ""
    
    combined_text = (vendor + " " + raw_text).lower()
    
    # Meals keywords
    meal_keywords = ["restaurant", "cafe", "coffee", "food", "pizza", "burger", "chicken", "subway", 
                     "taco", "sushi", "diner", "bar", "pub", "bistro", "bakery", "grocery", "supermarket",
                     "whole foods", "trader joe's", "sprouts", "safeway", "albertsons", "target food"]
    
    # Travel keywords
    travel_keywords = ["uber", "lyft", "taxi", "hotel", "motel", "airbnb", "airline", "delta", "united",
                      "southwest", "parking", "gas station", "shell", "chevron", "exxon", "bp", "hertz",
                      "avis", "enterprise", "rental"]
    
    # Supplies keywords
    supplies_keywords = ["office", "staples", "home depot", "lowes", "amazon", "best buy", "electronic",
                        "computer", "software", "printer", "papermate", "pen", "notebook", "supply"]
    
    for keyword in meal_keywords:
        if keyword in combined_text:
            return "meals"
    
    for keyword in travel_keywords:
        if keyword in combined_text:
            return "travel"
    
    for keyword in supplies_keywords:
        if keyword in combined_text:
            return "supplies"
    
    return "other"


def _extract_line_items(raw_text: str) -> Optional[str]:
    """Extract line items from receipt text."""
    lines = raw_text.splitlines()
    items = []
    
    # Look for lines that have amount values (price indicators)
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        # Skip header/footer lines
        if any(keyword in line.lower() for keyword in ["total", "subtotal", "tax", "receipt", "invoice", "date", "vendor"]):
            continue
        
        # If line has amount pattern, likely an item
        if re.search(_AMOUNT_REGEX, line):
            items.append(line)
    
    if items:
        return "|".join(items[:10])  # Store up to 10 items
    
    return None


def _extract_invoice_number(raw_text: str) -> Optional[str]:
    """Extract invoice/receipt number from text."""
    patterns = [
        r"(?:invoice|receipt)[\s#:]*(\d+)",
        r"(?:inv|rec|ref)[\s#:]*(\d+)",
        r"#\s*(\d{6,})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def extract_fields_from_image(image_path: str) -> dict:
    """Run OCR on receipt image and return raw text.
    
    All field extraction is handled by parse_receipt_text_with_llm in llm_parser.py.
    This function focuses only on image preprocessing and OCR.
    """
    image = Image.open(Path(image_path)).convert("RGB")
    
    # Attempt preprocessing for better OCR (fallback to simple if cv2 unavailable)
    try:
        preprocessed = _preprocess_image(image.copy())
    except Exception:
        # If advanced preprocessing fails, use simple version
        preprocessed = _preprocess_image_simple(image.copy())
    
    # OCR full image with enhanced config
    raw_text = pytesseract.image_to_string(preprocessed, config="--oem 3 --psm 6")
    
    # Optional: OCR regions for better accuracy on specific parts
    top_text = _ocr_region(preprocessed, top_percent=0.0, bottom_percent=0.25)
    bottom_text = _ocr_region(preprocessed, top_percent=0.65, bottom_percent=1.0)
    
    # Combine all OCR results
    combined_text = raw_text + "\n" + top_text + "\n" + bottom_text
    
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    
    # Simple vendor guess (optional, for reference)
    vendor_guess = _guess_vendor(lines)

    return {
        "doc_type": "receipt",
        "raw_text": combined_text,  # Comprehensive OCR text
        "vendor_guess": vendor_guess,  # Optional simple guess
        "description": f"Receipt from {vendor_guess}" if vendor_guess else "Receipt",
    }


def validate_totals(extracted: dict) -> Optional[str]:
    """Validate extracted totals and return a mismatch reason when invalid."""
    required = ("vendor", "date", "subtotal", "tax", "total")
    if any(extracted.get(field) in (None, "") for field in required):
        return "missing_fields"

    try:
        subtotal = float(extracted["subtotal"])
        tax = float(extracted["tax"])
        total = float(extracted["total"])
    except (TypeError, ValueError):
        return "missing_fields"

    diff = abs((subtotal + tax) - total)
    if diff > 0.05:
        return f"total_mismatch: subtotal+tax={subtotal + tax:.2f}, total={total:.2f}, diff={diff:.2f}"

    return None


def debug_extract(image_path: str) -> None:
    """Helper function for debugging OCR extraction. Prints vendor, date, totals, and first 20 OCR lines."""
    print(f"\n{'='*60}")
    print(f"DEBUG_EXTRACT: {image_path}")
    print(f"{'='*60}\n")
    
    extracted = extract_fields_from_image(image_path)
    
    print(f"Vendor: {extracted.get('vendor')}")
    print(f"Date: {extracted.get('date')}")
    print(f"Subtotal: {extracted.get('subtotal')}")
    print(f"Tax: {extracted.get('tax')}")
    print(f"Total: {extracted.get('total')}")
    print(f"Category: {extracted.get('category')}")
    print(f"Confidence: {extracted.get('confidence_overall')}")
    print(f"\nFirst 20 OCR lines:")
    print(f"{'-'*60}")
    
    raw_text = extracted.get('raw_text', '')
    lines = raw_text.splitlines()
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:2d}: {line}")
    
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")
    
    print(f"\n{'='*60}\n")
