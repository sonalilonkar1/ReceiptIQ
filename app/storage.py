"""Storage for extracted receipts with pending review status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


STORAGE_DIR = Path(__file__).parent.parent / "data" / "receipts_db"
RECEIPTS_FILE = STORAGE_DIR / "receipts.json"


def _ensure_storage_dir() -> None:
    """Ensure storage directory exists."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _load_all_receipts() -> dict:
    """Load all receipts from storage."""
    _ensure_storage_dir()
    if not RECEIPTS_FILE.exists():
        return {}
    with open(RECEIPTS_FILE) as f:
        return json.load(f)


def _save_all_receipts(receipts: dict) -> None:
    """Save all receipts to storage."""
    _ensure_storage_dir()
    with open(RECEIPTS_FILE, "w") as f:
        json.dump(receipts, f, indent=2)


def save_receipt(image_name: str, extracted_data: dict) -> None:
    """Save extracted receipt data."""
    receipts = _load_all_receipts()
    receipts[image_name] = extracted_data
    _save_all_receipts(receipts)


def get_receipt(image_name: str) -> Optional[dict]:
    """Get a single receipt by image name."""
    receipts = _load_all_receipts()
    return receipts.get(image_name)


def get_all_receipts() -> dict:
    """Get all receipts."""
    return _load_all_receipts()


def get_pending_review_receipts() -> list[dict]:
    """Get all receipts pending review, sorted by confidence (lowest first)."""
    receipts = _load_all_receipts()
    pending = [
        {"image_name": name, **data}
        for name, data in receipts.items()
        if data.get("pending_review", False)
    ]
    # Sort by confidence (lower = more fields missing)
    pending.sort(key=lambda x: x.get("confidence_overall", 0))
    return pending


def update_receipt(image_name: str, updates: dict) -> None:
    """Update a receipt with new data."""
    receipts = _load_all_receipts()
    if image_name not in receipts:
        raise ValueError(f"Receipt {image_name} not found")
    
    # Update the fields
    receipts[image_name].update(updates)
    
    # Recalculate pending_review status
    missing_fields = []
    if not receipts[image_name].get("vendor"):
        missing_fields.append("vendor")
    if not receipts[image_name].get("date"):
        missing_fields.append("date")
    if receipts[image_name].get("total") is None:
        missing_fields.append("total")
    if receipts[image_name].get("subtotal") is None:
        missing_fields.append("subtotal")
    if receipts[image_name].get("tax") is None:
        missing_fields.append("tax")
    
    receipts[image_name]["pending_review"] = bool(missing_fields)
    receipts[image_name]["missing_fields"] = missing_fields if missing_fields else None
    
    _save_all_receipts(receipts)


def delete_receipt(image_name: str) -> None:
    """Delete a receipt."""
    receipts = _load_all_receipts()
    if image_name in receipts:
        del receipts[image_name]
        _save_all_receipts(receipts)
