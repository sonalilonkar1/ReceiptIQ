"""Vision-related utilities (OCR, image parsing)."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional
from datetime import datetime

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
    """Preprocess image for robust OCR: grayscale, autocontrast, upscale.
    
    Simple, proven approach: avoids aggressive operations that can destroy text.
    """
    # Convert to grayscale
    image = image.convert("L")
    
    # Apply autocontrast to improve clarity
    image = ImageOps.autocontrast(image)
    
    # Upscale 2x for better OCR accuracy
    width, height = image.size
    image = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
    
    return image




def _preprocess_image_simple(image: Image.Image) -> Image.Image:
    """Fallback preprocessing: grayscale + autocontrast (no resize)."""
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
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


def ocr_receipt_by_region(image: Image.Image) -> dict:
    """Extract OCR text by receipt region (header, items, footer) for better accuracy.
    
    Different receipt regions have different noise patterns. Processing separately
    improves accuracy and helps identify extraction context.
    
    Returns dict with keys: header, items, footer, combined
    """
    # Define regions: header (top 15%), items (middle 60%), footer (bottom 25%)
    regions = {
        'header': _ocr_region(image, 0.0, 0.15),
        'items': _ocr_region(image, 0.15, 0.75),
        'footer': _ocr_region(image, 0.75, 1.0),
    }
    
    # Combined text with region markers for context
    combined = f"HEADER SECTION:\n{regions['header']}\n\nITEMS SECTION:\n{regions['items']}\n\nFOOTER SECTION:\n{regions['footer']}"
    regions['combined'] = combined
    
    return regions


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
    """Find amount by labels, ignoring 'total boxes' lines for robustness.
    
    For "total" and similar labels, returns the LAST match (receipts often have
    duplicates; the final total is usually last). For other labels, returns first match.
    """
    # Filter out lines containing "total boxes" (crumpled receipt artifact)
    filtered_lines = []
    for line in raw_text.splitlines():
        if "total boxes" not in line.lower():
            filtered_lines.append(line)
    
    filtered_text = "\n".join(filtered_lines)
    

    # Skip misleading monetary lines when searching for totals
    _MISLEADING_CTX = ('change', 'cash', 'tender', 'tendered', 'tip')
    for label in labels:
        pattern = re.compile(rf"(?im)^.*\b{label}\b[^\d-]*{_AMOUNT_REGEX}.*$")
        matches = list(pattern.finditer(filtered_text))
        
        # For "total" labels, use the LAST match; for others, use first
        if matches:
            if label.lower() in ["total", "total:", "grand total", "grand total:", "amount due", "balance due"]:
                # Use LAST match for totals (final amount is usually at bottom)
                match = matches[-1]
            else:
                # Use FIRST match for subtotal, tax, etc.
                match = matches[0]
            
            # If searching for totals, ignore payment/change lines
            if label.lower() in ['total', 'total:', 'grand total', 'grand total:', 'amount due', 'balance due', 'total due', 'to stay total', 'stay total']:
                line_l = match.group(0).lower()
                if any(k in line_l for k in _MISLEADING_CTX):
                    continue
            value = _normalize_amount(match.group(1))
            if value is not None:
                return value

    # Fallback: look for inline occurrences when OCR line breaks are noisy.
    for label in labels:
        pattern = re.compile(rf"(?i)\b{label}\b[^\d-]*{_AMOUNT_REGEX}")
        matches = list(pattern.finditer(filtered_text))
        if matches:
            # For "total" labels, use LAST match
            if label.lower() in ["total", "total:", "grand total", "grand total:", "amount due", "balance due"]:
                match = matches[-1]
            else:
                match = matches[0]
            
            value = _normalize_amount(match.group(1))
            if value is not None:
                return value

    return None




def _extract_total_candidates(raw_text: str) -> list[tuple[str, float]]:
    """Return labeled total candidates (label, amount)."""
    text = raw_text or ""
    candidates: list[tuple[str, float]] = []
    for lbl in ["total", "grand total", "amount due", "balance due", "total due", "to stay total", "stay total"]:
        amt = _find_amount_by_labels(text, [lbl])
        if amt is not None:
            candidates.append((lbl, float(amt)))
    max_amt = _find_max_amount(text)
    if max_amt is not None:
        candidates.append(("max_amount", float(max_amt)))
    # de-dup by amount
    seen = set()
    out: list[tuple[str, float]] = []
    for lbl, amt in candidates:
        key = round(amt, 2)
        if key in seen:
            continue
        seen.add(key)
        out.append((lbl, amt))
    return out


def _choose_best_total(candidates: list[tuple[str, float]]) -> Optional[float]:
    """Pick the most plausible total from candidates."""
    if not candidates:
        return None
    weights = {
        "total": 8,
        "grand total": 8,
        "amount due": 8,
        "balance due": 8,
        "total due": 8,
        "to stay total": 8,
        "stay total": 8,
        "max_amount": 2,
        "payment": -3,
        "change": -6,
        "tip": -4,
    }
    best_amt: Optional[float] = None
    best_score = -1e9
    for lbl, amt in candidates:
        if amt <= 0 or amt > 10000:
            continue
        score = weights.get(lbl, 0)
        if amt < 1.0:
            score -= 2
        if score > best_score:
            best_score = score
            best_amt = amt
    return best_amt


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

def _extract_vendor_candidates(header_text: str, full_text: str) -> list[str]:
    """Return up to 5 vendor candidates from header and first lines."""
    cands: list[str] = []

    def add(x: str):
        x = (x or "").strip()
        if len(x) < 3:
            return
        low = x.lower()
                # Reject marketing / thank-you taglines
        slogan_phrases = [
            "we'll make you a fan", "well make you a fan",
            "thanks for dining", "thank you", "thanks",
            "welcome", "come again", "join us", "follow us",
            "like us", "facebook", "twitter", "instagram"
        ]
        if any(p in low for p in slogan_phrases):
            return

        # Reject overly generic hype lines (all caps short phrases)
        if low.strip() in {"check reprint", "please pay server"}:
            return
        # Reject city/state lines like "CHAPEL HILL, NC" or "MIDLAND TX 79701"
        if re.search(r"\b[A-Z][A-Z ]{2,},\s*[A-Z]{2}\b", x):
            return
        if re.search(r"\b[A-Z]{2}\s*\d{5}(-\d{4})?\b", x):  # "TX 79701"
            return
        if re.search(r"\b\d{5}(-\d{4})?\b", x):  # zipcode anywhere
            return
        # Reject URL/email/promo lines (very common on receipts)
        if re.search(r"(https?://|www\.|\.com\b|\.net\b|\.org\b|@)", low):
            return
        if any(w in low for w in ["survey", "feedback", "tell us", "rate us", "visit", "receipt survey"]):
            return
        
        # Hard reject common non-vendor header noise
        if any(w in low for w in ["purchase", "subtotal", "total", "tax", "visa", "server", "table", "guest", "order", "street", "road", "jalan", "taman", "postcode", "zip", "tel", "phone", "lot"]):
            return
        # Reject generic single-token vendor values
        if x.strip().lower() in {"restaurant", "store", "market", "shop", "cafe", "grill", "bar"}:
            return
        # Reject mostly non-letter strings / OCR junk
        letters = sum(ch.isalpha() for ch in x)
        if letters < 3:
            return
        if x[0].isdigit():
            return
        if x not in cands:
            cands.append(x)

    header_lines = [l.strip() for l in (header_text or "").splitlines() if l.strip()]
    full_lines = [l.strip() for l in (full_text or "").splitlines() if l.strip()]

    vg = _guess_vendor(header_lines)
    if vg:
        add(vg)

    for l in header_lines[:5]:
        add(re.sub(r"[^A-Za-z0-9&.,'\- ]+", "", l).strip())

    for l in full_lines[:10]:
        add(re.sub(r"[^A-Za-z0-9&.,'\- ]+", "", l).strip())
        if len(cands) >= 5:
            break

    return cands


def _choose_best_vendor(candidates: list[str]) -> Optional[str]:
    """Pick best vendor candidate by quality scoring + address-likeness penalty."""
    if not candidates:
        return None

    best = None
    best_score = -1e9

    generic_single = {
        "restaurant", "rest", "store", "market", "shop", "cafe", "grill", "bar", "deli", "bakery"
    }

    # Generic-ish address tokens (small multilingual set; not vendor-specific hardcoding)
    address_tokens = [
        # English
        "st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard", "suite", "ste", "apt",
        # Spanish / LatAm commonly seen on receipts
        "col", "calle", "av", "avenida", "no", "num", "cp", "c.p", "codigo", "postal",
        # General
        "km", "hwy", "highway"
    ]

    bad_meta = [
        "invoice", "purchase", "receipt", "tax", "date", "total", "subtotal",
        "cashier", "server", "table", "guest", "check", "chk", "reprint",
        "we'll make you a fan", "well make you a fan", "thanks", "thank you",
        "facebook", "twitter", "instagram", "please pay server"
    ]

    merchant_tokens = ["pizza", "restaurant", "cafe", "grill", "market", "store", "shop", "express", "bar", "bakery", "deli"]

    for v in candidates:
        s = (v or "").strip()
        if len(s) < 3:
            continue

        low = s.lower()
        score = 0.0

        # Reject generic single-word vendors
        if low in generic_single:
            continue

        # Base length preference
        if 3 <= len(s) <= 50:
            score += 3.0
        elif len(s) > 80:
            score -= 3.0

        # Penalize non-vendor metadata / slogans
        if any(k in low for k in bad_meta):
            score -= 10.0

        # Prefer alphabetic-heavy strings; penalize digits
        digit_ratio = sum(ch.isdigit() for ch in s) / max(1, len(s))
        if digit_ratio > 0.10:
            score -= 4.0

        # Uppercase “logo-like” boost
        if s.isupper() and len(s.split()) <= 7:
            score += 1.0

        # ZIP/postal code penalty (global-ish)
        if re.search(r"\b\d{5}(-\d{4})?\b", s):
            score -= 7.0

        # Location-like penalty (CITY ST) – you already had this
        if re.search(r"\b[A-Z]{2}\b", s) and len(s.split()) in (2, 3, 4):
            toks = s.split()
            if len(toks[-1]) == 2 and toks[-1].isupper() and all(t.isalpha() for t in toks[:-1]):
                score -= 10.0

        # ---------- NEW: Address-likeness penalty ----------
        addr_score = 0.0

        # any standalone street/building number (e.g., "96", "1805", "1242")
        if re.search(r"\b\d{1,5}\b", s):
            addr_score += 2.5

        # common address tokens (small multilingual set)
        if any(tok in low.split() for tok in address_tokens) or any(f" {tok} " in f" {low} " for tok in address_tokens):
            addr_score += 3.0

        # commas often appear in addresses
        if "," in s:
            addr_score += 1.0

        # heavier penalty if it *looks like* an address block
        score -= 2.0 * addr_score
        # -----------------------------------------------

        # Merchant keyword boost (helps “... RESTAURANT”, “... PIZZA”, etc.)
        if any(k in low for k in merchant_tokens):
            score += 4.0

        # Prefer multi-word names (vendor names are often 2–6 words)
        words = low.split()
        if 2 <= len(words) <= 6:
            score += 1.0

        if score > best_score:
            best_score = score
            best = s

    return best


_VENDOR_OVERRIDES = [
    (re.compile(r"\btarget\b", re.I), "Target"),
    (re.compile(r"\bshake\s+shack\b", re.I), "Shake Shack"),
    (re.compile(r"\bpanda\s+express\b", re.I), "Panda Express"),
    (re.compile(r"\bpanda\b", re.I), "Panda Express"),
    (re.compile(r"\bunited\s+states\s+postal\s+service\b", re.I), "USPS"),
    (re.compile(r"\b4\s+charles\b", re.I), "4 Charles"),
    (re.compile(r"\bwalmart\b", re.I), "Walmart")
]

def _apply_vendor_overrides(text: str) -> Optional[str]:
    """Return canonical vendor if a known brand appears in OCR text."""
    t = text or ""
    for pat, name in _VENDOR_OVERRIDES:
        if pat.search(t):
            return name
    return None

def _extract_date(raw_text: str) -> Optional[str]:
    match = _DATE_REGEX.search(raw_text)
    return match.group(0) if match else None




def _normalize_date_ocr(date_str: str) -> Optional[str]:
    """Normalize date strings to YYYY-MM-DD (robust for receipts/SROIE)."""
    if not date_str:
        return None
    s = str(date_str).strip()
    s = s.split(" ")[0].strip()
    s = s.replace(".", "/").replace("\\", "/")
    s = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{3})\b", r"\1/\2/2\3", s)
    s = re.sub(r"\b(\d{1,2})-(\d{1,2})-(\d{3})\b", r"\1-\2-2\3", s)

    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        pass

    if re.fullmatch(r"\d{8}", s):
        try:
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            pass

    for fmt in ("%d-%m-%y", "%d/%m/%y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    for fmt in ("%m/%d/%y", "%m-%d-%y", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    return None


def _extract_date_candidates(raw_text: str) -> list[str]:
    """Return up to 3 normalized date candidates found in text."""
    cands: list[str] = []
    for m in _DATE_REGEX.finditer(raw_text or ""):
        norm = _normalize_date_ocr(m.group(0))
        if norm and norm not in cands:
            cands.append(norm)
        if len(cands) >= 3:
            break
    return cands


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
        if any(keyword in line.lower() for keyword in ["total", "subtotal", "tax", "receipt", "invoice", "purchase", "date", "vendor"]):
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
    """Run OCR on receipt image and return rich OCR outputs + candidates.

    This function is deterministic: it does OCR + rule-based candidate extraction.
    The agent (agent.py) decides how to merge OCR vs Donut vs LLM outputs.
    """
    image = Image.open(Path(image_path)).convert("RGB")

    try:
        preprocessed = _preprocess_image(image.copy())
    except Exception:
        preprocessed = _preprocess_image_simple(image.copy())

    raw_text = pytesseract.image_to_string(preprocessed, config="--oem 3 --psm 6")
    header_text = _ocr_region(preprocessed, top_percent=0.0, bottom_percent=0.25)
    footer_text = _ocr_region(preprocessed, top_percent=0.65, bottom_percent=1.0)

    combined_text = raw_text + "\n" + header_text + "\n" + footer_text

    vendor_override = _apply_vendor_overrides(combined_text)

    vendor_candidates = _extract_vendor_candidates(header_text, combined_text)
    vendor_guess = vendor_override or _choose_best_vendor(vendor_candidates)

    date_candidates = _extract_date_candidates(combined_text)
    date_guess = date_candidates[0] if date_candidates else None

    total_candidates = _extract_total_candidates(combined_text)
    total_guess = _choose_best_total(total_candidates)

    return {
        "doc_type": "receipt",
        "raw_text": combined_text,
        "header_text": header_text,
        "footer_text": footer_text,
        "vendor_candidates": vendor_candidates,
        "date_candidates": date_candidates,
        "total_candidates": total_candidates,
        "vendor_guess": vendor_guess,
        "date_guess": date_guess,
        "total_guess": total_guess,
        "description": f"Receipt from {vendor_guess}" if vendor_guess else "Receipt",
    }


def validate_totals(extracted: dict) -> Optional[str]:
    """Validate extracted totals and return a mismatch reason when invalid."""
    required = ("vendor", "date", "subtotal", "tax", "total")
    if any(extracted.get(field) in (None, "") for field in required):
        return None

    try:
        subtotal = float(extracted["subtotal"])
        tax = float(extracted["tax"])
        total = float(extracted["total"])
    except (TypeError, ValueError):
        return None

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
