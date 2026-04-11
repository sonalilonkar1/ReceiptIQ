"""Agent orchestration module for ReceiptIQ."""

from __future__ import annotations

from typing import Optional

from app.tools.db import (
    add_flag,
    average_spend_per_period,
    check_expense_rules_violations,
    compare_spending_periods,
    create_reimbursement_batch,
    detect_anomalies,
    export_to_csv_format,
    find_by_amount_threshold,
    find_duplicates,
    find_missing_fields,
    find_receipts_with_keywords,
    get_audit_flags,
    get_document_by_id,
    get_recent_docs,
    get_reimbursement_summary,
    insert_document,
    spend_by_category,
    spend_by_vendor,
    verify_vendor,
)
from app.tools.vision import extract_fields_from_image, validate_totals
from app.tools.web import web_lookup

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    _PHI_MODEL = None  # Lazy load on first use
    _MISTRAL_MODEL = None  # Lazy load on first use
    _MISTRAL_TOKENIZER = None
    _PHI_TOKENIZER = None
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False


def _format_recent_docs(rows: list[tuple]) -> str:
    """Format recent documents for display.
    
    Shape: (doc_id, vendor, doc_date, currency, total, created_at)
    """
    if not rows:
        return "No receipts found in database."
    
    formatted = "**Recent Receipts:**\n\n"
    for doc_id, vendor, doc_date, currency, total, created_at in rows:
        total_str = f"{total:.2f}" if isinstance(total, (int, float)) else str(total)
        formatted += f"• Doc #{doc_id}: {vendor or 'Unknown'} | {doc_date or 'No date'} | {total_str} {currency}\n"
    
    return formatted.strip()


def _format_spend_by_vendor(rows: list[tuple]) -> str:
    """Format vendor spend summary for display.
    
    Shape: (vendor, sum_total)
    """
    if not rows:
        return "No vendor spend data found."
    
    formatted = "**Spend by Vendor:**\n\n"
    total_spend = sum(amount for _, amount in rows)
    
    for vendor, amount in rows:
        amount_str = f"{amount:.2f}" if isinstance(amount, (int, float)) else str(amount)
        pct = (amount / total_spend * 100) if total_spend > 0 else 0
        formatted += f"• {vendor or 'Unknown'}: ${amount_str} ({pct:.1f}%)\n"
    
    total_str = f"{total_spend:.2f}" if isinstance(total_spend, (int, float)) else str(total_spend)
    formatted += f"\n**Total Spend: ${total_str}**"
    
    return formatted.strip()


def _format_duplicates(rows: list[tuple]) -> str:
    """Format duplicate documents for display.
    
    Shape: (vendor, doc_date, total, count)
    """
    if not rows:
        return "No duplicate receipts found."
    
    formatted = "**Potential Duplicates Found:**\n\n"
    for vendor, doc_date, total, count in rows:
        total_str = f"{total:.2f}" if isinstance(total, (int, float)) else str(total)
        formatted += f"• {count} copies: {vendor or 'Unknown'} on {doc_date or 'No date'} | ${total_str}\n"
    
    return formatted.strip()


def _format_category_spending(rows: list[tuple], days: int = 30) -> str:
    """Format spending by category.
    
    Shape: (category, sum_total, count)
    """
    if not rows:
        return f"No expenses found in the last {days} days."
    
    formatted = f"**Spending by Category (Last {days} Days):**\n\n"
    total_spend = sum(amount for _, amount, _ in rows)
    
    for category, amount, count in rows:
        amount_str = f"{amount:.2f}" if isinstance(amount, (int, float)) else str(amount)
        pct = (amount / total_spend * 100) if total_spend > 0 else 0
        formatted += f"• {category or 'Other'}: ${amount_str} ({pct:.1f}%) - {count} transactions\n"
    
    total_str = f"{total_spend:.2f}" if isinstance(total_spend, (int, float)) else str(total_spend)
    formatted += f"\n**Total: ${total_str}**"
    
    return formatted.strip()


def _format_missing_fields(rows: list[dict]) -> str:
    """Format documents with missing fields."""
    if not rows:
        return "All documents have required fields."
    
    formatted = "**Documents with Missing Fields:**\n\n"
    for doc in rows:
        formatted += f"• Doc #{doc['doc_id']}: {doc['vendor'] or 'Unknown'} - Missing: {doc['missing_field']}\n"
    
    return formatted.strip()


def _format_threshold_results(rows: list[tuple]) -> str:
    """Format amount threshold search results."""
    if not rows:
        return "No receipts found matching the criteria."
    
    formatted = "**Receipts Matching Criteria:**\n\n"
    for doc_id, vendor, total, doc_date, created_at in rows:
        total_str = f"{total:.2f}" if isinstance(total, (int, float)) else str(total)
        formatted += f"• Doc #{doc_id}: {vendor or 'Unknown'} on {doc_date or 'N/A'} | ${total_str}\n"
    
    return formatted.strip()


def _format_rule_violations(violations: list[dict]) -> str:
    """Format expense rule violations."""
    if not violations:
        return "No expense rule violations found."
    
    formatted = "**Expense Rule Violations:**\n\n"
    total_overage = sum(v.get("overage", 0) for v in violations)
    
    for v in violations:
        total_str = f"{v['total']:.2f}" if isinstance(v['total'], (int, float)) else str(v['total'])
        overage_str = f"{v['overage']:.2f}" if isinstance(v['overage'], (int, float)) else str(v['overage'])
        formatted += f"• Doc #{v['doc_id']}: {v['vendor']} on {v['doc_date']} | ${total_str} (overage: ${overage_str})\n"
    
    total_overage_str = f"{total_overage:.2f}" if isinstance(total_overage, (int, float)) else str(total_overage)
    formatted += f"\n**Total Overage: ${total_overage_str}**"
    
    return formatted.strip()


def _format_spending_comparison(comparison: dict, period1: str, period2: str) -> str:
    """Format spending comparison between two periods."""
    if not comparison:
        return "No data available for comparison."
    
    formatted = f"**Spending Comparison: {period1} vs {period2}:**\n\n"
    
    for category, data in sorted(comparison.items()):
        p1 = data['period1']
        p2 = data['period2']
        change = data['change']
        change_pct = data['change_pct']
        
        p1_str = f"{p1:.2f}" if isinstance(p1, (int, float)) else str(p1)
        p2_str = f"{p2:.2f}" if isinstance(p2, (int, float)) else str(p2)
        change_str = f"{change:.2f}" if isinstance(change, (int, float)) else str(change)
        
        change_indicator = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        formatted += f"• {category or 'Other'}: ${p1_str} → ${p2_str} {change_indicator} ${change_str} ({change_pct:+.1f}%)\n"
    
    return formatted.strip()


def _format_reimbursement_summary(summary: dict) -> str:
    """Format reimbursement batch summary."""
    if not summary:
        return "No reimbursement data found."
    
    formatted = f"""📋 **Reimbursement Summary: {summary['batch_name']}**

**Period:** {summary['start_date']} to {summary['end_date']}
**Total Amount:** ${summary['total_amount']:.2f}
**Status:** {summary['status']}
**Document Count:** {summary['document_count']}

**Breakdown by Category:**
"""
    
    for category, amount in sorted(summary['category_breakdown'].items(), key=lambda x: x[1], reverse=True):
        amount_str = f"{amount:.2f}" if isinstance(amount, (int, float)) else str(amount)
        formatted += f"\n• {category}: ${amount_str}"
    
    return formatted.strip()


def _format_anomalies(anomalies: list[dict]) -> str:
    """Format detected anomalies."""
    if not anomalies:
        return "No anomalies detected. Your receipts look good!"
    
    formatted = "🚨 **Anomalies Detected:**\n\n"
    
    # Group by type
    by_type = {}
    for anomaly in anomalies:
        atype = anomaly.get("anomaly_type", "unknown")
        if atype not in by_type:
            by_type[atype] = []
        by_type[atype].append(anomaly)
    
    for atype, items in by_type.items():
        formatted += f"**{atype.replace('_', ' ').title()}:**\n"
        for item in items:
            if item.get("doc_id"):
                formatted += f"  • Doc #{item['doc_id']}: {item['vendor']} - {item['details']}\n"
            else:
                formatted += f"  • {item['details']}\n"
        formatted += "\n"
    
    return formatted.strip()


def _format_document(doc: dict) -> str:
    """Format a single document for display with all details."""
    if not doc:
        return "Document not found."
    
    doc_id = doc.get("doc_id", "?")
    doc_type = doc.get("doc_type", "Unknown").upper()
    vendor = doc.get("vendor", "Unknown Vendor")
    doc_date = doc.get("doc_date", "Unknown Date")
    currency = doc.get("currency", "USD")
    subtotal = doc.get("subtotal")
    tax = doc.get("tax")
    total = doc.get("total")
    confidence = doc.get("confidence")
    category = doc.get("category", "other")
    invoice_number = doc.get("invoice_number")
    reimbursable = doc.get("reimbursable", 0)
    created_at = doc.get("created_at", "")
    flags = get_audit_flags(doc_id)
    
    subtotal_str = f"{subtotal:.2f}" if isinstance(subtotal, (int, float)) else str(subtotal)
    tax_str = f"{tax:.2f}" if isinstance(tax, (int, float)) else str(tax)
    total_str = f"{total:.2f}" if isinstance(total, (int, float)) else str(total)
    confidence_str = f"{confidence*100:.0f}%" if isinstance(confidence, (int, float)) else str(confidence)
    reimbursable_str = "✓ Yes" if reimbursable else "✗ No"
    
    formatted = f"""📄 **Document #{doc_id} ({doc_type})**

**Vendor:** {vendor}
**Date:** {doc_date}
**Category:** {category.capitalize()}
**Invoice #:** {invoice_number or 'N/A'}
**Reimbursable:** {reimbursable_str}

**Amounts:**
  • Subtotal: {currency} {subtotal_str}
  • Tax: {currency} {tax_str}
  • **Total: {currency} {total_str}**

**Confidence:** {confidence_str}
**Created:** {created_at}"""
    
    if flags:
        formatted += "\n\n⚠️ **Audit Flags:**\n"
        for flag in flags:
            formatted += f"  • {flag['flag_type'].upper()}: {flag['detail']}\n"
    
    return formatted.strip()


def _load_phi_model():
    """Lazy load Phi-3.5-mini model for intent routing."""
    global _PHI_MODEL, _PHI_TOKENIZER
    if _PHI_MODEL is None:
        _PHI_TOKENIZER = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            trust_remote_code=True
        )
        _PHI_MODEL = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
    return _PHI_MODEL, _PHI_TOKENIZER


def _load_mistral_model():
    """Lazy load Mistral-7B-Instruct model for complex reasoning."""
    global _MISTRAL_MODEL, _MISTRAL_TOKENIZER
    if _MISTRAL_MODEL is None:
        _MISTRAL_TOKENIZER = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            trust_remote_code=True
        )
        _MISTRAL_MODEL = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
    return _MISTRAL_MODEL, _MISTRAL_TOKENIZER


def _get_intent_with_llm(user_text: str) -> str:
    """Use Phi-3.5-mini to determine user intent intelligently."""
    if not _LLM_AVAILABLE:
        return _route_intent_keywords(user_text)
    
    try:
        model, tokenizer = _load_phi_model()
        
        # Phi-3.5 uses instruction format
        intent_instruction = f"""You are an expense management AI assistant. Classify the user's intent into ONE of these categories:
- recent: List recent receipts
- spend_by_vendor: Analyze spending by vendor
- spending_by_category: Show spending by category (meals/travel/supplies)
- duplicates: Find duplicate receipts
- missing_fields: Find receipts with missing information
- threshold_search: Find receipts above/below amount
- rule_violations: Check expense rule violations (like lunch limit)
- compare_periods: Compare spending between time periods
- export_csv: Export receipts as CSV
- average_spend: Calculate average spending
- keyword_search: Search for receipts with keywords
- reimbursement: Create reimbursement summaries
- web_lookup: Convert currency or lookup vendor info
- anomalies: Detect suspicious/anomalous receipts

User request: {user_text}

Respond with ONLY the intent category (one word from the list above):"""
        
        inputs = tokenizer(intent_instruction, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with constraints
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.3,
            top_p=0.9,
        )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract intent from response
        response_text = response_text.lower().split("respond with only")[-1].strip()
        
        # Valid intents
        valid_intents = {
            "recent", "spend_by_vendor", "spending_by_category", "duplicates",
            "missing_fields", "threshold_search", "rule_violations", "compare_periods",
            "export_csv", "average_spend", "keyword_search", "reimbursement",
            "web_lookup", "anomalies"
        }
        
        # Find the intent in response
        for intent in valid_intents:
            if intent in response_text:
                return intent
        
        return "unknown"
    
    except Exception:
        # Fallback to keyword routing if LLM fails
        return _route_intent_keywords(user_text)


def _analyze_with_mistral(user_text: str, context: str) -> str:
    """Use Mistral-7B-Instruct for complex reasoning over context."""
    if not _LLM_AVAILABLE:
        return context
    
    try:
        model, tokenizer = _load_mistral_model()
        
        # Mistral uses [INST] format
        instruction = f"""You are an expert expense management assistant. 
User query: {user_text}

Database context:
{context}

Provide a clear, concise analysis based on the data above."""
        
        messages = [
            {"role": "user", "content": instruction}
        ]
        
        # Format for Mistral
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )
        
        analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return analysis.split("Assistant:")[-1].strip() if "Assistant:" in analysis else analysis.strip()
    
    except Exception:
        # Return context as-is if Mistral analysis fails
        return context


def _route_intent_keywords(user_text: str) -> str:
    """Fallback keyword-based routing when Phi-3.5-mini is unavailable."""
    text = user_text.lower()

    if "duplicate" in text:
        return "duplicates"
    if "category" in text or "meals" in text or "travel" in text or "supplies" in text:
        return "spending_by_category"
    if "missing" in text or "incomplete" in text:
        return "missing_fields"
    if ("over" in text or "exceed" in text or "limit" in text or "violation" in text) and ("25" in text or "lunch" in text):
        return "rule_violations"
    if ("compare" in text or "between" in text) and ("january" in text or "february" in text or "last month" in text):
        return "compare_periods"
    if "export" in text or "csv" in text or "table" in text:
        return "export_csv"
    if "average" in text or "per week" in text or "per month" in text:
        return "average_spend"
    if "$" in text and ("over" in text or "above" in text or "below" in text):
        return "threshold_search"
    if "parking" in text or "toll" in text or "find" in text:
        return "keyword_search"
    if "reimburs" in text or "batch" in text or "summary" in text:
        return "reimbursement"
    if "list" in text and "receipt" in text:
        return "recent"
    if any(token in text for token in ("spend", "spent", "vendor")):
        return "spend_by_vendor"
    if "convert" in text or "exchange rate" in text:
        return "web_lookup"
    return "unknown"


def _route_intent(user_text: str) -> str:
    """Route user intent using Phi-3.5-mini instruction-following model."""
    return _get_intent_with_llm(user_text)


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
        "category": extracted.get("category", "other"),
        "line_items": extracted.get("line_items"),
        "description": extracted.get("description"),
        "invoice_number": extracted.get("invoice_number"),
        "raw_text": extracted.get("raw_text"),
    }


def handle_message(user_text: str, file_path: Optional[str] = None) -> dict:
    """Handle user requests with either file extraction or keyword-based routing."""
    import re
    citations: list[str] = []
    debug: dict = {"mode": "file" if file_path else "text"}
    
    # Check if user is asking for a specific document by ID
    doc_id_match = re.search(r"(?:doc|document|receipt)\s*#?(\d+)", user_text, re.IGNORECASE)
    if doc_id_match and not file_path:
        doc_id = int(doc_id_match.group(1))
        doc = get_document_by_id(doc_id)
        if doc:
            response = _format_document(doc)
            citations.append(f"DB:documents.doc_id={doc_id}")
            debug["intent"] = "document_lookup"
            debug["doc_id"] = doc_id
            return {"response": response, "citations": citations, "debug": debug}
        else:
            response = f"Document #{doc_id} not found in database."
            debug["intent"] = "document_lookup"
            debug["doc_id"] = doc_id
            return {"response": response, "citations": citations, "debug": debug}

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
        response = _format_recent_docs(rows)
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "spend_by_vendor":
        rows = spend_by_vendor()
        debug["rows"] = rows
        response = _format_spend_by_vendor(rows)
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "duplicates":
        rows = find_duplicates()
        debug["rows"] = rows
        response = _format_duplicates(rows)
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "web_lookup":
        web_result = web_lookup(user_text)
        debug["web_result"] = web_result
        
        if web_result.get("type") == "currency_conversion":
            if "error" in web_result:
                response = f"❌ {web_result['error']}"
            else:
                response = f"💱 **Currency Conversion:**\n{web_result['note']}\nExchange rate: {web_result['rate']}"
        else:
            response = web_result.get("note", "Web lookup completed.")
        
        citations.append("WEB:currency_api")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "spending_by_category":
        rows = spend_by_category(days=30)
        debug["rows"] = rows
        response = _format_category_spending(rows, days=30)
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "missing_fields":
        rows = find_missing_fields()
        debug["rows"] = rows
        response = _format_missing_fields(rows)
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "threshold_search":
        # Parse amount from user text
        import re as re_module
        match = re_module.search(r"\$?(\d+(?:\.\d{2})?)", user_text)
        min_amount = float(match.group(1)) if match else 100.0
        
        # Check for date range
        days = 90
        if "90" in user_text:
            days = 90
        elif "week" in user_text:
            days = 7
        elif "month" in user_text:
            days = 30
        
        rows = find_by_amount_threshold(min_amount=min_amount, days=days)
        debug["rows"] = rows
        response = _format_threshold_results(rows)
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "rule_violations":
        violations = check_expense_rules_violations(rule_name="lunch_limit")
        debug["violations"] = violations
        response = _format_rule_violations(violations)
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "compare_periods":
        # For simplicity, compare last month with previous month
        response = "Spending comparison requires specific date ranges. Please specify periods (e.g., 'Jan vs Feb', 'Last month vs this month')."
        debug["intent"] = "compare_periods"
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "average_spend":
        period = "week" if "week" in user_text.lower() else "month"
        avg = average_spend_per_period(period=period)
        debug["average"] = avg
        period_str = f"per {period}" if period else "per month"
        avg_str = f"{avg:.2f}" if isinstance(avg, (int, float)) else str(avg)
        response = f"**Average Spend {period_str.title()}:** ${avg_str}"
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "export_csv":
        # Parse days if specified
        days = None
        import re as re_module
        match = re_module.search(r"(\d+)\s*(?:day|month|week)", user_text)
        if match:
            val = int(match.group(1))
            if "month" in user_text:
                days = val * 30
            elif "week" in user_text:
                days = val * 7
            else:
                days = val
        
        csv_content = export_to_csv_format(days=days)
        response = f"```csv\n{csv_content}\n```"
        citations.append("DB:documents")
        debug["format"] = "csv"
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "keyword_search":
        # Extract keywords from query
        keywords = user_text.replace("find", "").replace("receipt", "").replace("with", "").split()
        keywords = [k.strip() for k in keywords if k.strip() and k.lower() not in ["the", "a", "an", "or", "and"]]
        
        if keywords:
            rows = find_receipts_with_keywords(keywords)
            debug["keywords"] = keywords
            debug["rows"] = rows
            
            formatted = f"**Receipts containing: {', '.join(keywords)}:**\n\n"
            if rows:
                for doc in rows:
                    formatted += f"• Doc #{doc['doc_id']}: {doc['vendor']} on {doc['doc_date']} | ${doc['total']:.2f}\n"
            else:
                formatted = f"No receipts found containing: {', '.join(keywords)}"
            
            response = formatted
        else:
            response = "Please specify keywords to search for (e.g., 'Find parking receipts')."
        
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "reimbursement":
        # Create or summarize reimbursement batch
        if "create" in user_text.lower() or "batch" in user_text.lower():
            # For now, create a default batch for current month
            response = "Reimbursement batch creation requires date ranges. Please specify dates (e.g., 'Feb 1-15')."
        else:
            response = "Use 'Create reimbursement batch Feb 1-15' to generate a summary for expense reporting."
        
        debug["intent"] = "reimbursement"
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "anomalies" or ("suspicious" in user_text.lower() or "anomal" in user_text.lower()):
        anomalies = detect_anomalies()
        debug["anomalies"] = anomalies
        response = _format_anomalies(anomalies)
        citations.append("DB:documents")
        return {"response": response, "citations": citations, "debug": debug}

    if intent == "vendor_verification" or "verify" in user_text.lower():
        # Extract vendor name from query
        import re as re_module
        vendor_match = re_module.search(r"(?:verify|check)\s+(?:vendor|company)?\s*(.+?)(?:\s+|$)", user_text, re.IGNORECASE)
        vendor_name = vendor_match.group(1).strip() if vendor_match else None
        
        if vendor_name:
            result = verify_vendor(vendor_name)
            debug["vendor_result"] = result
            
            status = "✓ Verified" if result["verified"] else "✗ Not Verified"
            response = f"""🏢 **Vendor Verification: {result['vendor']}**

**Status:** {status}
**Type:** {result['type']}
**Website:** {result.get('website', 'N/A')}
**Confidence:** {result.get('confidence', 0)*100:.0f}%"""
            
            if result.get("note"):
                response += f"\n**Note:** {result['note']}"
            
            citations.append("WEB:vendor_registry")
        else:
            response = "Please specify which vendor to verify (e.g., 'Verify McDonald's')"
        
        return {"response": response, "citations": citations, "debug": debug}

    response = (
        "I can help with:\\n"
        "• Process receipt images\\n"
        "• List recent receipts\\n"
        "• Analyze spending by vendor or category\\n"
        "• Find duplicate receipts\\n"
        "• Search by amount threshold\\n"
        "• Check expense rule violations\\n"
        "• Detect anomalies in expenses\\n"
        "• Verify vendor information\\n"
        "• Export receipts as CSV\\n"
        "• Currency conversion\\n"
        "• And more! Try asking specific questions about your expenses."
    )
    return {"response": response, "citations": citations, "debug": debug}
