"""Vision-related utilities (OCR, image parsing)."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

from PIL import Image
import pytesseract


_DATE_REGEX = re.compile(
    r"\b(?:"
    r"\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}"
    r"|\d{4}[\/-]\d{1,2}[\/-]\d{1,2}"
    r")\b"
)
_AMOUNT_REGEX = r"(-?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})|-?\$?\s*\d+(?:\.\d{2}))"


def _normalize_amount(raw: str) -> Optional[float]:
    cleaned = raw.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_amount_by_labels(raw_text: str, labels: list[str]) -> Optional[float]:
    for label in labels:
        pattern = re.compile(rf"(?im)^.*\b{label}\b[^\d-]*{_AMOUNT_REGEX}.*$")
        for match in pattern.finditer(raw_text):
            value = _normalize_amount(match.group(1))
            if value is not None:
                return value

    # Fallback: look for inline occurrences when OCR line breaks are noisy.
    for label in labels:
        pattern = re.compile(rf"(?i)\b{label}\b[^\d-]*{_AMOUNT_REGEX}")
        match = pattern.search(raw_text)
        if match:
            value = _normalize_amount(match.group(1))
            if value is not None:
                return value

    return None


def _guess_vendor(lines: list[str]) -> Optional[str]:
    for line in lines[:8]:
        candidate = re.sub(r"[^A-Za-z0-9&.,'\- ]+", "", line).strip()
        if len(candidate) < 3:
            continue
        lowered = candidate.lower()
        if any(token in lowered for token in ("receipt", "invoice", "tax", "date", "total", "subtotal")):
            continue
        if re.search(r"\d", candidate) and len(candidate.split()) <= 2:
            continue
        return candidate
    return None


def _extract_date(raw_text: str) -> Optional[str]:
    match = _DATE_REGEX.search(raw_text)
    return match.group(0) if match else None


def extract_fields_from_image(image_path: str) -> dict:
    """Run OCR and extract key receipt fields into a dictionary."""
    image = Image.open(Path(image_path)).convert("RGB")
    raw_text = pytesseract.image_to_string(image)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    vendor = _guess_vendor(lines)
    date = _extract_date(raw_text)
    subtotal = _find_amount_by_labels(raw_text, ["subtotal", "sub total"])
    tax = _find_amount_by_labels(raw_text, ["tax", "vat", "gst", "sales tax"])
    total = _find_amount_by_labels(raw_text, ["total", "amount due", "balance due", "grand total"])

    extracted = {
        "doc_type": "receipt",
        "vendor": vendor,
        "date": date,
        "currency": "USD",
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "raw_text": raw_text,
    }

    score = 0
    score += 1 if vendor else 0
    score += 1 if date else 0
    score += 1 if subtotal is not None else 0
    score += 1 if tax is not None else 0
    score += 1 if total is not None else 0
    extracted["confidence_overall"] = round(score / 5.0, 2)

    return extracted


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
