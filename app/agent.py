"""Agent orchestration module for ReceiptIQ."""

from __future__ import annotations

from typing import Optional

from app.tools.db import (
    add_flag,
    find_duplicates,
    get_recent_docs,
    insert_document,
    spend_by_vendor,
)
from app.tools.vision import extract_fields_from_image, validate_totals
from app.tools.web import web_lookup


def _route_intent(user_text: str) -> str:
    """Simple keyword routing for MVP; can be replaced by an LLM planner later."""
    text = user_text.lower()

    if "duplicate" in text:
        return "duplicates"
    if "list" in text and "receipt" in text:
        return "recent"
    if any(token in text for token in ("spend", "spent", "vendor")):
        return "spend_by_vendor"
    if "convert" in text or "exchange rate" in text:
        return "web_lookup"
    return "unknown"


def _to_document_payload(extracted: dict) -> dict:
    """Map extracted vision keys to the DB schema payload."""
    return {
        "doc_type": extracted.get("doc_type", "receipt"),
        "vendor": extracted.get("vendor"),
        "doc_date": extracted.get("date"),
        "currency": extracted.get("currency", "USD"),
        "subtotal": extracted.get("subtotal"),
        "tax": extracted.get("tax"),
        "total": extracted.get("total"),
        "confidence": extracted.get("confidence_overall"),
        "raw_text": extracted.get("raw_text"),
    }


def handle_message(user_text: str, file_path: Optional[str] = None) -> dict:
    """Handle user requests with either file extraction or keyword-based routing."""
    citations: list[str] = []
    debug: dict = {"mode": "file" if file_path else "text"}

    if file_path:
        extracted = extract_fields_from_image(file_path)
        db_payload = _to_document_payload(extracted)
        doc_id = insert_document(db_payload)
        citations.append(f"DB:documents.doc_id={doc_id}")

        mismatch = validate_totals(extracted)
        debug["validation"] = mismatch

        flagged = False
        if mismatch:
            add_flag(doc_id, "totals_validation", mismatch)
            citations.append(f"DB:audit_flags.doc_id={doc_id}")
            flagged = True

        vendor = extracted.get("vendor") or "unknown vendor"
        date = extracted.get("date") or "unknown date"
        total = extracted.get("total")
        total_text = f"{total:.2f}" if isinstance(total, (int, float)) else str(total)

        response = (
            f"Processed receipt for {vendor} on {date}. "
            f"Total: {total_text}. Saved as doc_id={doc_id}."
        )
        if flagged:
            response += f" Audit flag added ({mismatch})."

        debug["intent"] = "file_ingest"
        debug["doc_id"] = doc_id
        debug["flagged"] = flagged
        return {"response": response, "citations": citations, "debug": debug}

    intent = _route_intent(user_text)
    debug["intent"] = intent

    if intent == "recent":
        rows = get_recent_docs()
        debug["rows"] = rows
        response = "Recent receipts fetched."
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "spend_by_vendor":
        rows = spend_by_vendor()
        debug["rows"] = rows
        response = "Vendor spend summary fetched."
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "duplicates":
        rows = find_duplicates()
        debug["rows"] = rows
        response = "Duplicate candidates fetched."
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "web_lookup":
        web_result = web_lookup(user_text)
        debug["web_result"] = web_result
        response = "Used web lookup stub for conversion/exchange request."
        citations.append("WEB:stub")
        return {"response": response, "citations": citations, "debug": debug}

    response = (
        "I can process a receipt image, list receipts, summarize spend by vendor, "
        "find duplicates, or run web lookup for conversion requests."
    )
    return {"response": response, "citations": citations, "debug": debug}
