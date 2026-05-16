"""Donut OCR-free model for receipt field extraction (fallback extractor)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Global model cache
_model_cache = {}

DONUT_CHECKPOINTS = {
    "sroie": "hf-tuner/donut-base-finetuned-sroie",
    "cord": "naver-clova-ix/donut-base-finetuned-cord-v2",
}

_ADDR_STOPWORDS = [
    "lot", "jalan", "taman", "klang", "selangor", "postcode", "zip",
    "street", "st", "road", "rd", "ave", "avenue", "blvd", "lane", "ln",
    "tel", "phone"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_donut_model(task: str = "sroie") -> tuple:
    """Load Donut model and processor (cached globally).
    
    Args:
        task: "sroie" (default) or "cord"
        
    Returns:
        Tuple of (processor, model)
    """
    if task in _model_cache:
        return _model_cache[task]
    
    checkpoint = DONUT_CHECKPOINTS.get(task, DONUT_CHECKPOINTS["sroie"])
    
    print(f">>> Loading Donut model from {checkpoint}...")
    processor = DonutProcessor.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    model.to(DEVICE)
    
    _model_cache[task] = (processor, model)
    return processor, model


import re
from datetime import datetime

def _normalize_date_donut(date_str: str) -> str | None:
    """
    Normalize receipt dates to YYYY-MM-DD.
    Handles formats commonly seen in SROIE/receipts:
      - dd-mm-yy, dd/mm/yy
      - dd-mm-yyyy, dd/mm/yyyy
      - mm/dd/yy, mm/dd/yyyy (fallback)
      - yyyy-mm-dd
    """
    if not date_str:
        return None

    s = str(date_str).strip()
    s = s.replace(".", "/").replace("\\", "/")
    s = re.sub(r"\s+", " ", s)

    # Fix 3-digit years like 8/22/023 -> 8/22/2023
    s = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{3})\b", r"\1/\2/2\3", s)
    s = re.sub(r"\b(\d{1,2})-(\d{1,2})-(\d{3})\b", r"\1-\2-2\3", s)

    # Fix time split "1 15:04 pm" -> "1:15:04 pm" (best-effort)
    s = re.sub(r"\b(\d{1,2})\s+(\d{2}:\d{2})\s*(am|pm)\b", r"\1:\2 \3", s, flags=re.I)
    # Already ISO
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        pass

    # If it looks like US-style with 4-digit year and '/', prefer month-first first
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", s):
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                dt = datetime.strptime(s, fmt)
                if dt.year < 1970:
                    dt = dt.replace(year=dt.year + 100)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue

# Try day-first formats first (common in SROIE: 13-05-18)
    for fmt in ("%d-%m-%y", "%d/%m/%y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            # Convert 2-digit year into 19xx/20xx consistently (python handles pivot, but we can be explicit)
            # If year < 1970, assume 2000s for receipts
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    # Fallback month-first (US-style)
    for fmt in ("%m/%d/%y", "%m-%d-%y", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    return None

def _parse_sroie_single_line(text: str) -> dict:
    """
    Parse Donut SROIE free-text outputs like:
    '22.90 13-05-18 99 speed mart s/b lot p.t. 2811, jalan ...'
    """
    out = {"vendor": None, "date": None, "total": None, "subtotal": None, "tax": None}

    t = " ".join(text.strip().split())
    # Repair compact/merged dates sometimes produced by Donut (e.g., 01-262019 or 01262019)
    t = re.sub(r"\b(\d{2})-(\d{2})(\d{4})\b", r"\1-\2-\3", t)
    t = re.sub(r"\b(\d{2})(\d{2})(\d{4})\b", r"\1/\2/\3", t)
    if not t:
        return out

    # total: first decimal number with 2 digits
    m_total = re.search(r"\b(\d+[.,]\d{2})\b", t)
    if m_total:
        out["total"] = float(m_total.group(1).replace(",", "."))

    # date: dd-mm-yy or dd/mm/yy or dd-mm-yyyy
    m_date = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", t)
    if m_date:
        out["date"] = m_date.group(1)  # normalize later in your existing normalize fn

        # vendor candidate: text after date
        after = t[m_date.end():].strip()

        # cut at address stopword
        lower = after.lower()
        cut = len(after)
        for w in _ADDR_STOPWORDS:
            m = re.search(rf"\b{re.escape(w)}\b", lower)
            if m:
                cut = min(cut, m.start())
        vendor = after[:cut].strip(" ,.-")

        # clean vendor: remove extra numbers/punctuation noise
        vendor = re.sub(r"\b\d{1,5}\b", "", vendor).strip()
        vendor = re.sub(r"\s+", " ", vendor).strip()
        # vendor should be short-ish
        if 2 <= len(vendor) <= 50:
            out["vendor"] = vendor

        # Extra guard: reject vendor strings that look like time/location/server metadata
        bad_tokens = [' am', ' pm', 'server', 'table', 'check', 'guest', 'cash', 'change']
        vlow = (out.get('vendor') or '').lower()
        if any(tok in vlow for tok in bad_tokens):
            out['vendor'] = None

    return out

def _parse_sroie_output(output_text: str) -> dict:
    """Parse Donut SROIE output to extract fields.
    
    SROIE format includes: store_name, address, phone, website, date, total, tax, subtotal
    Also handles invoice format with TAX INVOICE keywords.
    
    Robust parsing handles:
    - Standard Key: Value format
    - Plain values (especially uppercase store names)
    - Multiple field name variations
    - Currency formats (dots and commas)
    """
    result = {
        "vendor": None,
        "date": None,
        "total": None,
        "subtotal": None,
        "tax": None,
    }
    
    if not output_text or not output_text.strip():
        return result  # Empty output
    
    # Handle multiple line formats
    lines = output_text.split('\n')
    
    # Debug: log raw output to see what we're getting
    import sys
    print(f">>> Donut raw output ({len(output_text)} chars): {repr(output_text[:200])}", file=sys.stderr)
    
    # If output is mostly one line (common Donut SROIE), try single-line parsing first
    if "\n" not in output_text:
        sl = _parse_sroie_single_line(output_text)
        if sl.get("vendor"):
            result["vendor"] = sl["vendor"]
        if sl.get("date"):
            normalized = _normalize_date_donut(sl["date"])
            if normalized:
                result["date"] = normalized
        if sl.get("total") is not None:
            result["total"] = sl["total"]
        
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Pattern 1: Key: Value
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip().lower()
            value = parts[1].strip() if len(parts) > 1 else ""
            
            # Vendor matching
            if any(x in key for x in ['store_name', 'store name', 'company', 'merchant', 'shop', 'vendor']):
                if value and len(value) > 2:
                    result['vendor'] = value
            
            # Date matching  
            elif any(x in key for x in ['date', 'invoice_date', 'transaction_date', 'receipt_date']):
                if value:
                    normalized = _normalize_date_donut(value)
                    if normalized:
                        result['date'] = normalized
            
            # Total matching
            elif any(x in key for x in ['total', 'grand_total', 'final_total', 'amount_due', 'total_amount']):
                if value:
                    try:
                        clean_val = re.sub(r'[^\d.,]', '', value)
                        # Handle amounts like '1.062.60' (thousands separator as dot)
                        if clean_val.count('.') >= 2 and clean_val.count(',') == 0:
                            parts = clean_val.split('.')
                            clean_val = ''.join(parts[:-1]) + '.' + parts[-1]
                        clean_val = clean_val.replace(',', '.')
                        num_val = float(clean_val)
                        if 0 < num_val < 10000:
                            result['total'] = num_val
                    except (ValueError, AttributeError):
                        pass
            
            # Subtotal matching
            elif any(x in key for x in ['subtotal', 'sub_total', 'net_total', 'subtotal_amount']):
                if value:
                    try:
                        clean_val = re.sub(r'[^\d.,]', '', value)
                        # Handle amounts like '1.062.60' (thousands separator as dot)
                        if clean_val.count('.') >= 2 and clean_val.count(',') == 0:
                            parts = clean_val.split('.')
                            clean_val = ''.join(parts[:-1]) + '.' + parts[-1]
                        clean_val = clean_val.replace(',', '.')
                        num_val = float(clean_val)
                        if 0 < num_val < 10000:
                            result['subtotal'] = num_val
                    except (ValueError, AttributeError):
                        pass
            
            # Tax matching
            elif any(x in key for x in ['tax', 'total_tax', 'gst', 'vat', 'service_charge', 'tax_amount']):
                if value:
                    try:
                        clean_val = re.sub(r'[^\d.,]', '', value)
                        # Handle amounts like '1.062.60' (thousands separator as dot)
                        if clean_val.count('.') >= 2 and clean_val.count(',') == 0:
                            parts = clean_val.split('.')
                            clean_val = ''.join(parts[:-1]) + '.' + parts[-1]
                        clean_val = clean_val.replace(',', '.')
                        num_val = float(clean_val)
                        if 0 <= num_val < 1000:
                            result['tax'] = num_val
                    except (ValueError, AttributeError):
                        pass
        
        # Pattern 2: Look for plain values without colons (common in Donut output)
        # Store name usually uppercase and 3+ chars without numbers at start
        if i < 3 and ':' not in line and len(line) > 3 and result["vendor"] is None:
            # allow mixed-case, but avoid address-like lines
            low = line.lower()
            if not any(w in low for w in _ADDR_STOPWORDS) and not any(k in low for k in ["total", "tax", "date", "invoice"]):
                # avoid lines that are mostly numbers
                digit_ratio = sum(ch.isdigit() for ch in line) / max(1, len(line))
                if digit_ratio < 0.15:
                    result["vendor"] = line.strip()
                    print(f">>> Found vendor from plain line {i}: {line}", file=sys.stderr)
        
        # Pattern 3: Extract numbers ONLY with strong cues.
        # Avoid treating postcodes/addresses as totals or tax.
        if ':' not in line and any(c.isdigit() for c in line):
            low = line.lower()

            # Only set TOTAL if the line contains a total cue OR looks like a pure amount line
            total_cue = any(k in low for k in ["total", "amount due", "grand total", "balance due"])
            tax_cue = any(k in low for k in ["tax", "gst", "vat", "service"])

            # Prefer decimal amounts; ignore long integers (e.g., 41150)
            decs = re.findall(r"\b(\d+[.,]\d{2})\b", line)
            for a in decs:
                try:
                    amt = float(a.replace(",", "."))
                except ValueError:
                    continue

                if total_cue and result["total"] is None and 0.5 < amt < 10000:
                    result["total"] = amt
                    print(f">>> Found total (cue) from line {i}: {amt}", file=sys.stderr)
                elif tax_cue and result["tax"] is None and 0 <= amt < 1000:
                    result["tax"] = amt
                    print(f">>> Found tax (cue) from line {i}: {amt}", file=sys.stderr)
    
    return result


def _parse_cord_output(output_text: str) -> dict:
    """Parse Donut CORD output to extract fields.
    
    CORD has different format but similar fields.
    """
    result = {
        "vendor": None,
        "date": None,
        "total": None,
        "subtotal": None,
        "tax": None,
    }
    
    # CORD format varies; look for key patterns
    lines = output_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
        
        key, value = line.split(':', 1)
        key = key.strip().lower()
        value = value.strip()
        
        # Try common field names
        if any(x in key for x in ['merchant', 'store', 'company', 'vendor']):
            result['vendor'] = value if value else None
        elif 'date' in key or 'time' in key:
            normalized = _normalize_date_donut(value)
            result['date'] = normalized
        elif any(x in key for x in ['total', 'amount']):
            try:
                clean_val = re.sub(r'[^\d.]', '', value)
                result['total'] = float(clean_val) if clean_val else None
            except (ValueError, AttributeError):
                pass
    
    return result


def extract_fields_donut(
    image_path: Optional[str] = None,
    pil_image: Optional[Image.Image] = None,
    task: str = "sroie",
) -> dict:
    """Extract receipt fields using Donut model (OCR-free).
    
    Direct image processing - more reliable than OCR for certain receipt formats.
    
    Args:
        image_path: Path to image file (str or Path)
        pil_image: PIL Image object (alternative to image_path)
        task: "sroie" (default) or "cord"
        
    Returns:
        Dict with keys: vendor, date, total, subtotal, tax, raw, extraction_source, confidence
    """
    # Validate inputs
    if image_path is None and pil_image is None:
        return {
            "vendor": None,
            "date": None,
            "total": None,
            "subtotal": None,
            "tax": None,
            "raw": "No image provided",
            "extraction_source": "donut",
            "confidence": 0.0,
        }
    
    # Load image if needed
    if pil_image is None:
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {
                "vendor": None,
                "date": None,
                "total": None,
                "subtotal": None,
                "tax": None,
                "raw": f"Failed to load image: {str(e)}",
                "extraction_source": "donut",
                "confidence": 0.0,
            }
    
    # Load model (cached)
    processor, model = _load_donut_model(task)
    
    # Prepare input
    try:
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values
    except Exception as e:
        return {
            "vendor": None,
            "date": None,
            "total": None,
            "subtotal": None,
            "tax": None,
            "raw": f"Failed to process image: {str(e)}",
            "extraction_source": "donut",
            "confidence": 0.0,
        }
    
    # Generate output
    try:
        with torch.no_grad():
            pixel_values = pixel_values.to(DEVICE)
            outputs = model.generate(pixel_values, max_length=1024)
        
        output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        return {
            "vendor": None,
            "date": None,
            "total": None,
            "subtotal": None,
            "tax": None,
            "raw": f"Donut extraction failed: {str(e)}",
            "extraction_source": "donut",
            "confidence": 0.0,
        }
    
    # Parse output based on task
    if task == "sroie":
        parsed = _parse_sroie_output(output_text)
    elif task == "cord":
        parsed = _parse_cord_output(output_text)
    else:
        parsed = {
            "vendor": None,
            "date": None,
            "total": None,
            "subtotal": None,
            "tax": None,
        }
    
    # Post-process to clean values
    if parsed.get("vendor"):
        # Remove stray IDs or numbers from start
        vendor = parsed["vendor"].strip()
        # If starts with 7+ digit ID, try to remove it
        if re.match(r'^\d{7,}.*', vendor):
            parts = vendor.split()
            if len(parts) > 1 and len(parts[0]) > 6:
                # First part is likely ID, skip it
                vendor = ' '.join(parts[1:])
        parsed["vendor"] = vendor if len(vendor) > 2 else parsed.get("vendor")
    
    # Validate totals are reasonable (<$10k)
    for field in ["total", "subtotal", "tax"]:
        if parsed.get(field) is not None and parsed[field] > 10000:
            parsed[field] = None  # Likely OCR error
    
    # Calculate confidence: count how many fields were extracted
    extracted_count = sum(1 for v in [parsed.get("vendor"), parsed.get("date"), parsed.get("total")] if v is not None)
    confidence = extracted_count / 3.0  # 0.33, 0.66, or 1.0
    
    return {
        "vendor": parsed.get("vendor"),
        "date": parsed.get("date"),
        "total": parsed.get("total"),
        "subtotal": parsed.get("subtotal"),
        "tax": parsed.get("tax"),
        "raw": output_text,
        "extraction_source": "donut",
        "confidence": confidence,
    }
