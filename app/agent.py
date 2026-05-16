"""Agent orchestration module for ReceiptIQ."""

from __future__ import annotations

from typing import Optional
import json
import re
import time
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler('receiptiq.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# DEBUG LOGGING FLAG - Set to True to enable logs, False to disable
DEBUG_LOGS = True

def log_receipt(message: str, doc_id: int = None):
    """Log receipt-related messages when DEBUG_LOGS is enabled."""
    if DEBUG_LOGS:
        prefix = f"[Receipt #{doc_id}]" if doc_id else "[Receipt]"
        print(f"📋 {prefix} {message}", flush=True)
        logger.info(f"{prefix} {message}")


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
    get_vendor_category,
    insert_document,
    list_pending_receipts,
    spend_by_category,
    spend_by_category_calendar,
    spend_by_vendor,
    verify_vendor,
    _connect,
)
from app.tools.vision import extract_fields_from_image, validate_totals, _classify_category, _extract_invoice_number, _extract_line_items
from app.tools.web import web_lookup, verify_vendor_online, normalize_vendor_name
from app.tools.llm_parser import parse_receipt_text_with_llm
from app.tools.donut import extract_fields_donut

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


# LLM Chaining Configuration
USE_LLM_CHAINING = False  # Set to True to enable prompt chaining pipeline for specific intents
CHAINABLE_INTENTS = {"spend_by_vendor", "anomalies"}  # Intents that support chaining

# Model Mode Configuration
MODEL_MODE = "phi_only"  # Options: "phi_only" (use formatters) or "phi+mistral" (Mistral writes output)
# - "phi_only": Phi routes intent → use deterministic formatter functions
# - "phi+mistral": Phi routes intent → Mistral writes natural language response for DB intents

# Prompt Caching Configuration
PROMPT_CACHE_ENABLED = True  # Enable/disable prompt template caching in memory
# When enabled: Prompts loaded once and reused; significant performance improvement
# When disabled: Prompts reloaded from disk on every use (slower but useful for testing)

# Cached prompt templates (loaded on first use, reused if cache enabled)
_SYSTEM_PROMPT = None
_PLANNER_PROMPT = None
_VERIFIER_PROMPT = None

# Timing metrics for cache performance
_PROMPT_CACHE_TIMING = {
    "system": {"load_time_ms": 0, "cache_hits": 0, "cache_misses": 0},
    "planner": {"load_time_ms": 0, "cache_hits": 0, "cache_misses": 0},
    "verifier": {"load_time_ms": 0, "cache_hits": 0, "cache_misses": 0},
}


def _load_prompt(filename: str) -> Optional[str]:
    """Load a prompt file from app/prompts/ directory."""
    try:
        prompt_path = Path(__file__).parent / "prompts" / filename
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                return f.read()
    except Exception:
        pass
    return None


def _get_system_prompt() -> Optional[str]:
    """Get cached system prompt or load it. Returns None if caching disabled or file not found."""
    global _SYSTEM_PROMPT, _PROMPT_CACHE_TIMING
    
    if not PROMPT_CACHE_ENABLED:
        # Cache disabled: reload from disk every time
        _PROMPT_CACHE_TIMING["system"]["cache_misses"] += 1
        return _load_prompt("system.txt")
    
    # Cache enabled: load once and reuse
    if _SYSTEM_PROMPT is None:
        start = time.time()
        _SYSTEM_PROMPT = _load_prompt("system.txt")
        load_ms = (time.time() - start) * 1000
        _PROMPT_CACHE_TIMING["system"]["load_time_ms"] = round(load_ms, 2)
        _PROMPT_CACHE_TIMING["system"]["cache_misses"] += 1
    else:
        _PROMPT_CACHE_TIMING["system"]["cache_hits"] += 1
    
    return _SYSTEM_PROMPT


def _get_planner_prompt() -> Optional[str]:
    """Get cached planner prompt or load it. Returns None if caching disabled or file not found."""
    global _PLANNER_PROMPT, _PROMPT_CACHE_TIMING
    
    if not PROMPT_CACHE_ENABLED:
        # Cache disabled: reload from disk every time
        _PROMPT_CACHE_TIMING["planner"]["cache_misses"] += 1
        return _load_prompt("planner.txt")
    
    # Cache enabled: load once and reuse
    if _PLANNER_PROMPT is None:
        start = time.time()
        _PLANNER_PROMPT = _load_prompt("planner.txt")
        load_ms = (time.time() - start) * 1000
        _PROMPT_CACHE_TIMING["planner"]["load_time_ms"] = round(load_ms, 2)
        _PROMPT_CACHE_TIMING["planner"]["cache_misses"] += 1
    else:
        _PROMPT_CACHE_TIMING["planner"]["cache_hits"] += 1
    
    return _PLANNER_PROMPT


def _get_verifier_prompt() -> Optional[str]:
    """Get cached verifier prompt or load it. Returns None if caching disabled or file not found."""
    global _VERIFIER_PROMPT, _PROMPT_CACHE_TIMING
    
    if not PROMPT_CACHE_ENABLED:
        # Cache disabled: reload from disk every time
        _PROMPT_CACHE_TIMING["verifier"]["cache_misses"] += 1
        return _load_prompt("verifier.txt")
    
    # Cache enabled: load once and reuse
    if _VERIFIER_PROMPT is None:
        start = time.time()
        _VERIFIER_PROMPT = _load_prompt("verifier.txt")
        load_ms = (time.time() - start) * 1000
        _PROMPT_CACHE_TIMING["verifier"]["load_time_ms"] = round(load_ms, 2)
        _PROMPT_CACHE_TIMING["verifier"]["cache_misses"] += 1
    else:
        _PROMPT_CACHE_TIMING["verifier"]["cache_hits"] += 1
    
    return _VERIFIER_PROMPT


def get_prompt_cache_metrics() -> dict:
    """Get cache performance metrics for all prompts.
    
    Returns dict with load times and hit/miss counts for each prompt type.
    Useful for benchmarking cache effectiveness.
    """
    return {
        "cache_enabled": PROMPT_CACHE_ENABLED,
        "metrics": dict(_PROMPT_CACHE_TIMING),
        "timestamp": time.time(),
    }


def reset_prompt_cache():
    """Reset all cached prompts and timing metrics. Used for testing."""
    global _SYSTEM_PROMPT, _PLANNER_PROMPT, _VERIFIER_PROMPT, _PROMPT_CACHE_TIMING
    _SYSTEM_PROMPT = None
    _PLANNER_PROMPT = None
    _VERIFIER_PROMPT = None
    _PROMPT_CACHE_TIMING = {
        "system": {"load_time_ms": 0, "cache_hits": 0, "cache_misses": 0},
        "planner": {"load_time_ms": 0, "cache_hits": 0, "cache_misses": 0},
        "verifier": {"load_time_ms": 0, "cache_hits": 0, "cache_misses": 0},
    }

def _wrap_response(response: str, citations: list[str], debug: dict, success: bool = True, error: str | None = None) -> dict:
    """Standardize response object for benchmark compatibility."""
    out = {"response": response, "citations": citations, "debug": debug, "success": success}
    # Prefer debug latency if set
    out["latency_ms"] = debug.get("latency_ms")
    if error:
        out["error"] = error
        out["success"] = False
    return out

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


def _get_intent_with_ollama(user_text: str) -> str:
    """Use Ollama (Phi-3.5-mini) to determine user intent via HTTP API."""
    try:
        import requests
        
        # Valid intents
        valid_intents = [
            "recent", "spend_by_vendor", "spending_by_category", "weekly_summary",
            "monthly_summary", "pending_receipts", "duplicates", "missing_fields",
            "threshold_search", "rule_violations", "compare_periods", "export_csv",
            "average_spend", "keyword_search", "reimbursement", "web_lookup", "anomalies", "vendor_verification"
        ]
        
        intent_instruction = f"""You are an expense management AI assistant. Classify the user's intent into ONE of these categories:
- recent: List recent receipts
- spend_by_vendor: Analyze spending by vendor
- spending_by_category: Show spending by category (meals/travel/supplies)
- weekly_summary: Show weekly spending summary
- monthly_summary: Show monthly spending summary
- pending_receipts: List receipts with incomplete or missing data
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
- vendor_verification: Verify vendor information
User request: {user_text}

Respond with ONLY the intent category (one word from the list above):"""
        
        # Call Ollama via HTTP API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3.5",
                "prompt": intent_instruction,
                "stream": False,
                "temperature": 0.3,
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return _route_intent_keywords(user_text)
        
        try:
            response_data = response.json()
        except Exception:
            response_data = None

        if isinstance(response_data, dict):
            response_text = str(response_data.get("response", "")).lower().strip()
        else:
            response_text = (response.text or "").lower().strip()
        
        # Extract intent from response
        for intent in valid_intents:
            if intent in response_text:
                return intent
        
        return _route_intent_keywords(user_text)
    
    except Exception:
        # Fallback to keyword routing if Ollama fails or unavailable
        return _route_intent_keywords(user_text)


def _get_intent_with_llm(user_text: str) -> str:
    """Route user intent - tries Ollama first, then falls back to keyword routing."""
    # Try Ollama (HTTP API) first - much faster than loading model locally
    try:
        return _get_intent_with_ollama(user_text)
    except Exception:
        pass
    
    # Fallback to keyword routing
    return _route_intent_keywords(user_text)


def _analyze_with_mistral(user_text: str, context: str) -> str:
    """Use Ollama Mistral to analyze context - HTTP API approach."""
    try:
        import requests
        
        instruction = f"""You are an expert expense management assistant. 
User query: {user_text}

Database context:
{context}

Provide a clear, concise analysis based on the data above."""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": instruction,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
    
    except Exception:
        pass
    
    # Return context as-is if analysis fails
    return context


def _plan_with_phi(user_text: str, tool_schema: str) -> Optional[dict]:
    """Use Ollama Phi to plan the query - HTTP API approach."""
    if not _get_planner_prompt():
        return None
    
    try:
        import requests
        
        planner_prompt = _get_planner_prompt()
        
        instruction = f"""{planner_prompt}

User query: {user_text}

Tool schema:
{tool_schema}"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3.5",
                "prompt": instruction,
                "stream": False,
                "temperature": 0.3,
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return None

        try:
            data = response.json()
        except Exception:
            data = None

        if isinstance(data, dict):
            response_text = str(data.get("response", ""))
        else:
            response_text = response.text or ""
        
        # Extract JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            plan = json.loads(json_str)
            return plan
        
        return None
    
    except Exception:
        return None


def _write_answer_with_mistral(user_text: str, db_results: str) -> Optional[str]:
    """Use Ollama Mistral to write the final answer - HTTP API approach."""
    try:
        import requests
        
        instruction = f"""You are ReceiptIQ, an expense management assistant. 
User query: {user_text}

Database results:
{db_results}

Provide a clear, concise answer based on the data above. Always cite document IDs (doc_id) when referencing receipts. Never hallucinate data."""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": instruction,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
            except Exception:
                result = None

            if isinstance(result, dict):
                return str(result.get("response", "")).strip()
            else:
                return (response.text or "").strip()
    
    except Exception:
        pass
    
    return None


def _verify_answer_with_phi(draft_answer: str, tool_results: str) -> Optional[dict]:
    """Use Ollama Phi to verify the answer - HTTP API approach."""
    verifier_prompt = _get_verifier_prompt()
    if not verifier_prompt:
        return None
    
    try:
        import requests
        
        instruction = f"""{verifier_prompt}

Draft Answer: {draft_answer}

Tool Output:
{tool_results}"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3.5",
                "prompt": instruction,
                "stream": False,
                "temperature": 0.3,
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return None

        try:
            data = response.json()
        except Exception:
            data = None

        if isinstance(data, dict):
            response_text = str(data.get("response", ""))
        else:
            response_text = response.text or ""
        
        # Extract JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            verification = json.loads(json_str)
            return verification
        
        return None
    
    except Exception:
        return None


def _rewrite_with_mistral(formatted_output: str, user_query: str, intent: str) -> Optional[str]:
    """Use Ollama Mistral to rewrite formatter output as natural language - HTTP API approach."""
    try:
        import requests
        
        instruction = f"""You are ReceiptIQ, an expense management assistant. 
User asked: {user_query}
Intent: {intent}

Current formatted response:
{formatted_output}

Rewrite this response in natural, conversational language. 
IMPORTANT: Keep ALL document IDs (doc_id: X) and citations exactly as they appear.
Make it more engaging and easier to understand while preserving all data accuracy.
Keep formatting clean and organized."""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": instruction,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
            except Exception:
                result = None

            if isinstance(result, dict):
                rewritten = str(result.get("response", "")).strip()
            else:
                rewritten = (response.text or "").strip()
            
            # Ensure citations are preserved
            if "doc_id" in formatted_output and "doc_id" not in rewritten:
                # Fallback if Mistral drops citations
                rewritten += f"\n\n_Data source: {formatted_output.split('DB:')[-1] if 'DB:' in formatted_output else 'Database'}_"
            
            return rewritten if rewritten else formatted_output
    
    except Exception:
        pass
    
    # Fallback to original formatted output on error
    return formatted_output


def _route_intent_keywords(user_text: str) -> str:
    """Fallback keyword-based routing when Ollama is unavailable."""
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
    """Map extracted vision keys to the DB schema payload.
    
    Computes is_pending flag if any critical fields (vendor/date/total) are missing.
    Overrides category if vendor exists in vendor_profiles.
    """
    vendor = extracted.get("vendor")
    doc_date = extracted.get("date")
    total = extracted.get("total")
    
    # Compute missing critical fields
    missing_keys = []
    if not vendor:
        missing_keys.append("vendor")
    if not doc_date:
        missing_keys.append("date")
    if total is None or total == 0:
        missing_keys.append("total")
    
    is_pending = 1 if missing_keys else 0
    
    # Default category
    category = extracted.get("category", "other")
    
    # Override category if vendor exists in vendor_profiles (learned associations)
    if vendor:
        learned_category = get_vendor_category(vendor)
        if learned_category:
            category = learned_category
    
    return {
        "doc_type": extracted.get("doc_type", "receipt"),
        "vendor": vendor,
        "doc_date": doc_date,
        "currency": extracted.get("currency", "USD"),
        "subtotal": extracted.get("subtotal"),
        "tax": extracted.get("tax"),
        "total": total,
        "confidence": extracted.get("confidence_overall"),
        "category": category,
        "line_items": extracted.get("line_items"),
        "description": extracted.get("description"),
        "invoice_number": extracted.get("invoice_number"),
        "raw_text": extracted.get("raw_text"),
        "is_pending": is_pending,
    }


def security_guard(user_text: str) -> Optional[dict]:
    """
    Detect prompt injection attacks and refuse malicious requests.
    
    Returns dict with refusal response if injection detected, None otherwise.
    Sets refused=True in debug field for benchmark detection.
    
    Detects attempts to:
    - Dump or modify database
    - Reveal system prompts/instructions
    - Bypass tool usage
    - Modify stored data
    - Escalate privileges
    - Execute arbitrary commands
    """
    text_lower = user_text.lower()
    
    # Injection patterns to detect
    db_dump_patterns = [
        r"drop\s+table",
        r"delete\s+from",
        r"dump\s+db",
        r"dump\s+database",
        r"export.*all",
        r"raw\s+database",
    ]
    
    system_prompt_patterns = [
        r"reveal.*system\s+prompt",
        r"show.*instructions",
        r"system\s+prompt",
        r"reveal.*you.*are",
        r"what.*are.*your.*instructions",
        r"ignore.*previous.*instructions",
        r"override.*system",
    ]
    
    tool_bypass_patterns = [
        r"do\s+not\s+use.*tool",
        r"skip.*tool",
        r"bypass.*tool",
        r"guess\s+the.*total",
        r"just\s+guess",
        r"hallucinate",
    ]
    
    data_modification_patterns = [
        r"modify.*stored",
        r"change.*total",
        r"update.*receipt",
        r"corrupt.*data",
        r"fake\s+\d+",
        r"adjust.*amount",
        r"pretend.*verified",
        r"spoof",
        r"fake.*vendor",
    ]
    
    privilege_patterns = [
        r"admin\s+privilege",
        r"escalate\s+privilege",
        r"show.*all\s+user",
        r"other\s+user.*data",
        r"unauthorized\s+access",
    ]
    
    command_patterns = [
        r"execute:",
        r"run\s+command",
        r"rm\s+-rf",
        r"chmod",
        r"shell\s+command",
    ]
    
    # Check for injection patterns
    all_patterns = (
        db_dump_patterns +
        system_prompt_patterns +
        tool_bypass_patterns +
        data_modification_patterns +
        privilege_patterns +
        command_patterns
    )
    
    import re
    detected_attack = None
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            detected_attack = pattern
            break
    
    if detected_attack:
        # Determine attack type for better response
        if any(re.search(p, text_lower) for p in db_dump_patterns):
            attack_type = "database manipulation"
            safe_alternative = "I can summarize your spending patterns, show recent transactions, or analyze spending by category instead."
        elif any(re.search(p, text_lower) for p in system_prompt_patterns):
            attack_type = "system prompt extraction"
            safe_alternative = "I'm designed to help you manage your receipts and expenses. I can answer questions about your spending or help you analyze your financial data."
        elif any(re.search(p, text_lower) for p in tool_bypass_patterns):
            attack_type = "tool bypass attempt"
            safe_alternative = "I use database tools to provide accurate information. I can show you verified data about your receipts and spending instead."
        elif any(re.search(p, text_lower) for p in data_modification_patterns):
            attack_type = "data modification attempt"
            safe_alternative = "I can read and analyze your receipts, but cannot modify stored data. I can help you review your receipts for errors."
        elif any(re.search(p, text_lower) for p in privilege_patterns):
            attack_type = "privilege escalation"
            safe_alternative = "I only have access to your own receipts and spending data. I can help you analyze your personal expense information."
        elif any(re.search(p, text_lower) for p in command_patterns):
            attack_type = "command injection"
            safe_alternative = "I'm a text-based assistant. I can process receipt images or answer questions about your expenses instead."
        else:
            attack_type = "malicious request"
            safe_alternative = "I can help you with your receipts and expense management. What would you like to know about your spending?"
        
        response = f"I cannot process that request ({attack_type}). {safe_alternative}"
        
        return {
            "response": response,
            "citations": [],
            "debug": {
                "refused": True,
                "attack_type": attack_type,
                "pattern_matched": detected_attack,
            }
        }
    
    return None


def _parse_receipt_with_llm_fallback(raw_text: str, model: str = "phi") -> dict:
    """Parse receipt OCR text with LLM, falling back to regex if LLM fails.
    
    Args:
        raw_text: OCR-extracted receipt text
        model: "phi" (default, faster) or "mistral" (more capable)
    
    Returns dict compatible with _to_document_payload:
    {
        "vendor": str or None,
        "date": str (YYYY-MM-DD) or None,
        "subtotal": float or None,
        "tax": float or None,
        "total": float or None,
        "currency": str,
        "category": str,
        "invoice_number": str or None,
        "line_items": str or None,
        "raw_text": str,
        "doc_type": "receipt",
        "confidence_overall": float,
        "llm_confidence": dict with per-field confidence scores
    }
    """
    # Primary: LLM parsing
    log_receipt("LLM Extraction - Preprocessing OCR text (fix errors, remove duplicates)")
    log_receipt("LLM Extraction - Calling LLM parser (using Phi - faster & reliable)")
    llm_result = parse_receipt_text_with_llm(raw_text, model="phi")
    
    # Extract values from LLM result (structure: {field: {value, confidence, evidence_line}})
    vendor = llm_result.get("vendor", {}).get("value")
    date = llm_result.get("date", {}).get("value")
    subtotal = llm_result.get("subtotal", {}).get("value")
    tax = llm_result.get("tax", {}).get("value")
    total = llm_result.get("total", {}).get("value")
    currency = llm_result.get("currency", {}).get("value") or "USD"
    category = llm_result.get("category", {}).get("value") or "other"
    invoice_number = llm_result.get("invoice_number", {}).get("value")
    line_items = llm_result.get("line_items", {}).get("value")
    
    # Collect per-field confidence scores from LLM
    llm_confidence = {
        "vendor": llm_result.get("vendor", {}).get("confidence", 0.0),
        "date": llm_result.get("date", {}).get("confidence", 0.0),
        "total": llm_result.get("total", {}).get("confidence", 0.0),
        "subtotal": llm_result.get("subtotal", {}).get("confidence", 0.0),
        "tax": llm_result.get("tax", {}).get("confidence", 0.0),
    }
    
    # Log what LLM extracted
    log_receipt(f"LLM Extracted: vendor='{vendor}' (conf:{llm_confidence['vendor']:.2f}), date='{date}' (conf:{llm_confidence['date']:.2f}), total={total} (conf:{llm_confidence['total']:.2f})")
    
    # Fallback to regex if critical fields are missing or LLM confidence too low
    if not vendor or llm_confidence.get("vendor", 0) < 0.5:
        # Try regex fallback for vendor
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        from app.tools.vision import _guess_vendor
        vendor_guess = _guess_vendor(lines)
        if vendor_guess and not vendor:
            log_receipt(f"Vendor Fallback: Using regex fallback. LLM='{vendor}' (conf:{llm_confidence['vendor']:.2f}) → Regex='{vendor_guess}'")
            vendor = vendor_guess
            llm_confidence["vendor"] = 0.3  # Lower confidence for regex fallback
        elif vendor and llm_confidence.get("vendor", 0) < 0.5:
            log_receipt(f"Vendor Fallback: LLM confidence too low. Keeping LLM value '{vendor}' (conf:{llm_confidence['vendor']:.2f})")
    
    if not date or llm_confidence.get("date", 0) < 0.5:
        # Try regex fallback for date
        from app.tools.vision import _extract_date
        date_guess = _extract_date(raw_text)
        if date_guess and not date:
            log_receipt(f"Date Fallback: Using regex fallback. LLM='{date}' (conf:{llm_confidence['date']:.2f}) → Regex='{date_guess}'")
            date = date_guess
            llm_confidence["date"] = 0.3
        elif date and llm_confidence.get("date", 0) < 0.5:
            log_receipt(f"Date Fallback: LLM confidence too low. Keeping LLM value '{date}' (conf:{llm_confidence['date']:.2f})")
    
    if not total or llm_confidence.get("total", 0) < 0.5:
        # CRITICAL: Before regex fallback, try calculating total from subtotal + tax
        if not total and subtotal and tax:
            calculated_total = subtotal + tax
            log_receipt(f"Total Fallback: Calculating from subtotal({subtotal}) + tax({tax}) = {calculated_total}")
            total = calculated_total
            llm_confidence["total"] = 0.7  # Moderate-high confidence for calculated value
        elif not total:
            # Try regex fallback for total
            from app.tools.vision import _find_amount_by_labels, _find_max_amount
            total_guess = _find_amount_by_labels(raw_text, ["total", "amount due", "balance due", "grand total"])
            if not total_guess:
                total_guess = _find_max_amount(raw_text)
            if total_guess:
                log_receipt(f"Total Fallback: Using regex fallback. LLM={total} (conf:{llm_confidence['total']:.2f}) → Regex={total_guess}")
                total = total_guess
                llm_confidence["total"] = 0.3
        elif total and llm_confidence.get("total", 0) < 0.5:
            log_receipt(f"Total Fallback: LLM confidence too low. Keeping LLM value {total} (conf:{llm_confidence['total']:.2f})")
    
    # Format line_items for database (string with items separated by |)
    line_items_str = None
    if line_items:
        if isinstance(line_items, list):
            line_items_str = "|".join(str(item) for item in line_items[:10])
        else:
            line_items_str = str(line_items)
    
    # Calculate overall confidence (weighted average of critical fields)
    overall_confidence = (
        llm_confidence.get("vendor", 0) * 0.25 +
        llm_confidence.get("date", 0) * 0.25 +
        llm_confidence.get("total", 0) * 0.25 +
        llm_confidence.get("subtotal", 0) * 0.125 +
        llm_confidence.get("tax", 0) * 0.125
    )
    
    # Log final values before returning
    log_receipt(f"Final Extracted Values: vendor='{vendor}', date='{date}', total={total}, subtotal={subtotal}, tax={tax}")
    log_receipt(f"Overall Confidence: {round(overall_confidence, 2):.2f}")
    log_receipt(f"Per-field Confidence: vendor={llm_confidence['vendor']:.2f}, date={llm_confidence['date']:.2f}, total={llm_confidence['total']:.2f}")
    
    return {
        "doc_type": "receipt",
        "vendor": vendor,
        "date": date,
        "currency": currency,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "category": category,
        "invoice_number": invoice_number,
        "line_items": line_items_str,
        "description": f"Receipt from {vendor}" if vendor else "Receipt",
        "raw_text": raw_text,
        "confidence_overall": round(overall_confidence, 2),
        "llm_confidence": llm_confidence,
        "llm_error": llm_result.get("error"),
    }

def handle_message(user_text: str, file_path: Optional[str] = None) -> dict:
    """Handle user requests with either file extraction or keyword-based routing."""
    import re
    citations: list[str] = []
    debug: dict = {"mode": "file" if file_path else "text"}
    
    # Security check: detect and refuse prompt injection attacks
    security_check = security_guard(user_text)
    if security_check:
        security_check["debug"].update(debug)  # Merge existing debug info
        return security_check
    
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
        # =====================
        # FILE INGEST PIPELINE
        # Goal: maximize extraction accuracy using field-fusion:
        #   OCR (deterministic candidates) -> Donut fallback -> Phi evidence-only fallback
        # =====================
        log_receipt("UPLOAD OPERATION STARTED - Processing image file")

        # 1) OCR + deterministic candidates
        ocr_result = extract_fields_from_image(file_path)
        raw_text = ocr_result.get("raw_text", "") or ""
        debug["ocr_chars"] = len(raw_text)

        vendor = ocr_result.get("vendor_guess")
        doc_date = ocr_result.get("date_guess")
        total = ocr_result.get("total_guess")

        extracted = {
            "doc_type": "receipt",
            "vendor": vendor,
            "date": doc_date,
            "total": total,
            "subtotal": None,
            "tax": None,
            "currency": "USD",
            "category": _classify_category(vendor, raw_text),
            "invoice_number": _extract_invoice_number(raw_text),
            "line_items": _extract_line_items(raw_text),
            "raw_text": raw_text,
            # track where each field came from
            "field_source": {
                "vendor": "ocr",
                "date": "ocr",
                "total": "ocr",
            },
        }

        log_receipt(f"OCR candidates: vendor={vendor!r}, date={doc_date!r}, total={total!r}")
        debug["vendor_candidates"] = ocr_result.get("vendor_candidates", [])
        debug["date_candidates"] = ocr_result.get("date_candidates", [])
        debug["total_candidates"] = ocr_result.get("total_candidates", [])

        # 2) Donut fallback (best for vendor/date; usually weak on totals)
        need_donut = (not extracted.get("vendor")) or (not extracted.get("date"))
        if need_donut:
            try:
                donut_result = extract_fields_donut(image_path=file_path, task="sroie")
                debug["donut_conf"] = donut_result.get("confidence")
                if not extracted.get("vendor") and donut_result.get("vendor"):
                    extracted["vendor"] = donut_result["vendor"]
                    extracted["field_source"]["vendor"] = "donut"
                if not extracted.get("date") and donut_result.get("date"):
                    extracted["date"] = donut_result["date"]
                    extracted["field_source"]["date"] = "donut"
                # only fill total from donut if OCR couldn't find one
                if (extracted.get("total") in (None, 0, "")) and donut_result.get("total") is not None:
                    extracted["total"] = donut_result["total"]
                    extracted["field_source"]["total"] = "donut"
                log_receipt(f"Donut fallback: vendor={donut_result.get('vendor')!r}, date={donut_result.get('date')!r}, total={donut_result.get('total')!r}")
            except Exception as e:
                log_receipt(f"Donut fallback failed: {e}")

        # If OCR total looks suspicious (e.g., came from CHANGE/CASH/TIP) and Donut has a total, override.
        try:
            tc = debug.get("total_candidates") or []
            tc_labels = [str(x[0]).lower() for x in tc if isinstance(x, (list, tuple)) and len(x) >= 2]
            if extracted.get("field_source", {}).get("total") == "ocr" and extracted.get("total") is not None:
                if any(lbl in tc_labels for lbl in ["change", "payment", "tip"]):
                    # use Donut total if present
                    if "donut_result" in locals() and donut_result and donut_result.get("total") is not None:
                        extracted["total"] = donut_result["total"]
                        extracted["field_source"]["total"] = "donut_override"
                        log_receipt(f"Total override: OCR looked like payment/change; using Donut total={donut_result.get('total')}")
        except Exception:
            pass

        # Lightweight vendor cleanup + optional online verification ONLY when vendor looks suspicious.
        try:
            vraw = extracted.get("vendor")
            if vraw:
                extracted["vendor"] = normalize_vendor_name(vraw)
                vlow = extracted["vendor"].lower()
                suspicious = (
                    len(extracted["vendor"]) < 3
                    or "expect more" in vlow
                    or "pay less" in vlow
                    or "purchase" in vlow
                    or vlow in {"united states", "united", "states"}
                )
                if suspicious:
                    ver = verify_vendor_online(extracted["vendor"], timeout_s=2.0)
                    debug["vendor_verify"] = {
                        "confidence": ver.get("confidence"),
                        "cached": ver.get("cached"),
                        "best": ver.get("best_guess_official"),
                        "error": ver.get("error"),
                    }
                    best = ver.get("best_guess_official") or {}
                    dom = (best.get("domain") or "").lower()
                    if "target.com" in dom:
                        extracted["vendor"] = "Target"
                        extracted["field_source"]["vendor"] = extracted["field_source"].get("vendor", "ocr") + "+webnorm"
                    elif "usps.com" in dom:
                        extracted["vendor"] = "USPS"
                        extracted["field_source"]["vendor"] = extracted["field_source"].get("vendor", "ocr") + "+webnorm"
                    elif "shakeshack.com" in dom:
                        extracted["vendor"] = "Shake Shack"
                        extracted["field_source"]["vendor"] = extracted["field_source"].get("vendor", "ocr") + "+webnorm"
        except Exception:
            pass

        # 3) Phi evidence-only fallback (only when still missing)

        missing = [k for k in ["vendor", "date", "total"] if not extracted.get(k)]
        if missing:
            log_receipt(f"Phi fallback triggered (missing={missing})")
            try:
                llm = parse_receipt_text_with_llm(raw_text, model="phi") or {}
                debug["phi_used"] = True

                def accept(field: str) -> Optional[str]:
                    node = llm.get(field) or {}
                    val = node.get("value")
                    ev = node.get("evidence_line") or ""
                    conf = node.get("confidence", 0.0)
                    # evidence must appear in OCR text (prevents hallucinations)
                    if not val:
                        return None
                    if ev and ev in raw_text:
                        # extra rule for totals: evidence line should include total-ish keyword
                        if field == "total" and not any(k in str(ev).lower() for k in ["total", "amount", "balance", "due", "grand"]):
                            return None
                        return val
                    # If no evidence provided, reject (safer)
                    return None

                if not extracted.get("vendor"):
                    v = accept("vendor")
                    if v:
                        extracted["vendor"] = v
                        extracted["field_source"]["vendor"] = "phi_evidence"
                if not extracted.get("date"):
                    d = accept("date")
                    if d:
                        extracted["date"] = d
                        extracted["field_source"]["date"] = "phi_evidence"
                if not extracted.get("total"):
                    t = accept("total")
                    if t is not None:
                        try:
                            extracted["total"] = float(str(t).replace(",", "").replace("$", "").strip())
                            extracted["field_source"]["total"] = "phi_evidence"
                        except Exception:
                            pass

            except Exception as e:
                log_receipt(f"Phi fallback failed: {e}")

        # 4) Final deterministic sanity (category depends on vendor)
        extracted["category"] = _classify_category(extracted.get("vendor"), raw_text)

        # Overall confidence = completeness score of critical fields
        filled = sum(1 for k in ["vendor", "date", "total"] if extracted.get(k))
        extracted["confidence_overall"] = round(filled / 3.0, 2)

        extraction_source = "+".join(sorted(set(extracted["field_source"].values())))
        extracted["extraction_source"] = extraction_source
        debug["field_source"] = extracted["field_source"]
        debug["extraction_source"] = extraction_source

        # 5) Save to DB
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

        is_pending = db_payload.get("is_pending", 0)
        missing_fields = []
        if not db_payload.get("vendor"):
            missing_fields.append("vendor")
        if not db_payload.get("doc_date"):
            missing_fields.append("date")
        if not db_payload.get("total"):
            missing_fields.append("total")

        vendor_disp = extracted.get("vendor") or "unknown vendor"
        date_disp = extracted.get("date") or "unknown date"
        total_disp = extracted.get("total")
        total_text = f"{total_disp:.2f}" if isinstance(total_disp, (int, float)) else str(total_disp)

        response = (
            f"Processed receipt for {vendor_disp} on {date_disp}. "
            f"Total: {total_text}. Saved as doc_id={doc_id}. "
            f"Pending: {'Yes' if is_pending else 'No'}."
        )
        if is_pending and missing_fields:
            response += f" Missing: {', '.join(missing_fields)}."
            response += " Open the 'Pending Receipts' tab to complete the data."
        if flagged:
            response += f" Audit flag added ({mismatch})."

        debug["intent"] = "file_ingest"
        debug["doc_id"] = doc_id
        debug["flagged"] = flagged
        debug["is_pending"] = is_pending
        debug["missing_fields"] = missing_fields
        return {"response": response, "citations": citations, "debug": debug}



    # Fast-path routing for common DB queries (avoids Ollama misclassification + improves latency)
    text_l = user_text.lower()

    # Weekly receipts query (saved this week / this week)
    if ("this week" in text_l or "saved this week" in text_l) and ("receipt" in text_l or "receipts" in text_l or "transaction" in text_l):
        intent = "weekly_summary"

    # Top spending categories this month
    elif ("top" in text_l and ("category" in text_l or "categories" in text_l)) and ("month" in text_l) and ("this" in text_l or "current" in text_l):
        intent = "spending_by_category"
        debug["top_k"] = 5
        debug["range"] = "this_month"

    elif ("subtotal" in text_l and "tax" in text_l and "total" in text_l) or ("consistent" in text_l and "subtotal" in text_l):
        intent = "validate_totals"
    elif ("top" in text_l and "category" in text_l) and ("this month" in text_l or "current month" in text_l):
        intent = "spending_by_category"
        debug["top_k"] = 5
        debug["range"] = "this_month"
    elif ("average" in text_l and "lunch" in text_l) and ("week" in text_l or "weekly" in text_l):
        intent = "avg_lunch_weekly"
    elif ("why" in text_l and "flag" in text_l) or ("explain" in text_l and "flag" in text_l):
        intent = "explain_flag"
    elif ("compare" in text_l or "between" in text_l) and ("january" in text_l and "february" in text_l):
        intent = "compare_spending_periods"
    elif "reimburs" in text_l and ("summary" in text_l or "totals" in text_l):
        intent = "reimbursement_summary"
    elif "draft" in text_l and "email" in text_l and "reimburs" in text_l:
        intent = "reimbursement_email"
    elif ("mark" in text_l and "reimburs" in text_l):
        intent = "mark_reimbursable"

    elif ("recent" in text_l and ("receipt" in text_l or "transaction" in text_l)) or ("show me all my receipts" in text_l):
        intent = "recent"
    elif "last month" in text_l and ("receipt" in text_l or "transaction" in text_l):
        intent = "recent"
    elif ("spend" in text_l or "spent" in text_l) and ("vendor" in text_l or "merchant" in text_l):
        intent = "spend_by_vendor"
    elif "category" in text_l and ("spend" in text_l or "spending" in text_l):
        intent = "spending_by_category"
    elif "duplicate" in text_l:
        intent = "duplicates"
    elif "anomal" in text_l or "unusual" in text_l or "suspicious" in text_l:
        intent = "anomalies"
    elif "pending" in text_l:
        intent = "pending_receipts"
    elif "weekly" in text_l:
        intent = "weekly_summary"
    elif "monthly" in text_l:
        intent = "monthly_summary"
    elif "convert" in text_l or "exchange rate" in text_l:
        intent = "web_lookup"
    elif "verify" in text_l and "vendor" in text_l:
        intent = "vendor_verification"
    else:
        intent = _route_intent(user_text)

    debug["intent"] = intent

    if intent == "recent":
        start_time = time.time()
        debug["routing_model"] = "phi"
        debug["writer_model"] = "formatter"
        
        use_mistral = MODEL_MODE == "phi+mistral" and _LLM_AVAILABLE
        
        if use_mistral:
            try:
                rows = get_recent_docs()
                formatted = _format_recent_docs(rows)
                response = _rewrite_with_mistral(formatted, user_text, "recent")
                debug["writer_model"] = "mistral"
                debug["models_used"] = ["Phi (router)", "Mistral (writer)"]
            except Exception:
                rows = get_recent_docs()
                response = _format_recent_docs(rows)
                debug["writer_model"] = "formatter"
                debug["models_used"] = ["Phi (router)"]
        else:
            rows = get_recent_docs()
            response = _format_recent_docs(rows)
            debug["models_used"] = ["Phi (router)"]
        
        debug["rows"] = rows if 'rows' in locals() else None
        debug["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        citations.append("DB:documents")
        return _wrap_response(response, citations, debug, success=True)

    if intent == "spend_by_vendor":
        start_time = time.time()
        debug["routing_model"] = "phi"
        debug["writer_model"] = "formatter"
        
        # Determine which mode to use
        use_chaining = USE_LLM_CHAINING and intent in CHAINABLE_INTENTS and _LLM_AVAILABLE
        use_mistral = MODEL_MODE == "phi+mistral" and _LLM_AVAILABLE and not use_chaining
        
        if use_chaining:
            # Use LLM chaining pipeline
            try:
                tool_schema = "DB Functions: spend_by_vendor() returns [(vendor, total_amount), ...]\nResponse styles: summary, detailed, formatted_table"
                plan = _plan_with_phi(user_text, tool_schema)
                debug["models_used"] = ["Phi (planner)"]
                
                if plan:
                    rows = spend_by_vendor()
                    db_results = f"Vendor spending data:\n{str(rows)}"
                    
                    draft_answer = _write_answer_with_mistral(user_text, db_results)
                    if draft_answer:
                        debug["models_used"].append("Mistral (writer)")
                        debug["writer_model"] = "mistral"
                        
                        verification = _verify_answer_with_phi(draft_answer, db_results)
                        debug["models_used"].append("Phi (verifier)")
                        
                        if verification and verification.get("is_supported"):
                            response = draft_answer
                            if verification.get("revised_answer"):
                                response = verification["revised_answer"]
                        else:
                            response = draft_answer if draft_answer else _format_spend_by_vendor(rows)
                    else:
                        response = _format_spend_by_vendor(rows)
                else:
                    rows = spend_by_vendor()
                    response = _format_spend_by_vendor(rows)
                
                debug["chain_type"] = "llm_chaining"
            except Exception as e:
                rows = spend_by_vendor()
                response = _format_spend_by_vendor(rows)
                debug["chain_error"] = str(e)
                debug["models_used"] = []
        
        elif use_mistral:
            # Use formatter + Mistral rewrite mode
            try:
                rows = spend_by_vendor()
                formatted = _format_spend_by_vendor(rows)
                response = _rewrite_with_mistral(formatted, user_text, "spend_by_vendor")
                debug["writer_model"] = "mistral"
                debug["models_used"] = ["Phi (router)", "Mistral (writer)"]
            except Exception as e:
                rows = spend_by_vendor()
                response = _format_spend_by_vendor(rows)
                debug["writer_model"] = "formatter"
                debug["models_used"] = ["Phi (router)"]
        
        else:
            # Use deterministic formatter only
            rows = spend_by_vendor()
            response = _format_spend_by_vendor(rows)
            debug["models_used"] = ["Phi (router)"]
        
        debug["rows"] = rows if 'rows' in locals() else None
        debug["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        citations.append("DB:documents")
        return _wrap_response(response, citations, debug, success=True)

    if intent == "duplicates":
        start_time = time.time()
        debug["routing_model"] = "phi"
        debug["writer_model"] = "formatter"
        
        use_mistral = MODEL_MODE == "phi+mistral" and _LLM_AVAILABLE
        
        if use_mistral:
            try:
                rows = find_duplicates()
                formatted = _format_duplicates(rows)
                response = _rewrite_with_mistral(formatted, user_text, "duplicates")
                debug["writer_model"] = "mistral"
                debug["models_used"] = ["Phi (router)", "Mistral (writer)"]
            except Exception:
                rows = find_duplicates()
                response = _format_duplicates(rows)
                debug["writer_model"] = "formatter"
                debug["models_used"] = ["Phi (router)"]
        else:
            rows = find_duplicates()
            response = _format_duplicates(rows)
            debug["models_used"] = ["Phi (router)"]
        
        debug["rows"] = rows if 'rows' in locals() else None
        debug["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        citations.append("DB:documents")
        return _wrap_response(response, citations, debug, success=True)

    if intent == "validate_totals":
        m = re.search(r"(?:doc|document|receipt)\s*#?(\d+)", user_text, re.IGNORECASE)
        doc_id = int(m.group(1)) if m else None
        if not doc_id:
            latest = get_recent_docs(limit=1)
            doc_id = int(latest[0][0]) if latest else None
        if not doc_id:
            return _wrap_response("No receipts found to validate.", citations, debug, success=True)

        doc = get_document_by_id(doc_id) or {}
        subtotal = doc.get("subtotal")
        tax = doc.get("tax")
        total = doc.get("total")

        mismatch = False
        if subtotal is not None and tax is not None and total is not None:
            try:
                expected = float(subtotal) + float(tax)
                mismatch = abs(expected - float(total)) > 0.02
            except Exception:
                mismatch = False

        citations.append(f"DB:documents.doc_id={doc_id}")
        citations.append(f"DB:audit_flags.doc_id={doc_id}")

        if mismatch:
            add_flag(doc_id, "totals_mismatch", f"subtotal+tax != total (subtotal={subtotal}, tax={tax}, total={total})")
            resp = f"⚠️ Totals mismatch for doc_id={doc_id}. I flagged it for review."
        else:
            resp = f"✅ Totals look consistent for doc_id={doc_id}."
        return _wrap_response(resp, citations, debug, success=True)

    if intent == "avg_lunch_weekly":
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT strftime('%Y-W%W', COALESCE(doc_date, substr(created_at,1,10))) AS wk,
                       COALESCE(SUM(total),0) AS wk_total
                FROM documents
                WHERE LOWER(COALESCE(category,'other')) = 'meals'
                GROUP BY wk
                ORDER BY wk DESC
                LIMIT 8
                """
            ).fetchall()
        totals = [float(r[1] or 0) for r in rows]
        avg = (sum(totals) / len(totals)) if totals else 0.0
        resp = f"🍔 **Average lunch/meals spend per week (last {len(totals)} weeks):** ${avg:.2f}"
        citations.append("DB:documents")
        debug["weekly_totals"] = rows
        return _wrap_response(resp, citations, debug, success=True)

    if intent == "reimbursement_summary":
        summary = get_reimbursement_summary(user_text)
        citations.append("DB:documents")
        return _wrap_response(summary, citations, debug, success=True)

    if intent == "reimbursement_email":
        summary = get_reimbursement_summary(user_text)
        body = (
            "Hi [Manager Name],\n\n"
            "Here is my reimbursable expense summary for the requested period:\n\n"
            f"{summary}\n\n"
            "Thanks,\n[Your Name]"
        )
        resp = "**Email Draft:**\n\n```\n" + body + "\n```"
        citations.append("DB:documents")
        return _wrap_response(resp, citations, debug, success=True)

    if intent == "mark_reimbursable":
        ids = [int(x) for x in re.findall(r"(?:doc|doc_id|document|receipt)\s*#?\s*(\d+)", user_text, re.IGNORECASE)]
        citations.append("DB:documents")
        if not ids:
            return _wrap_response("Tell me which doc IDs to mark reimbursable (e.g., 'Mark doc 12 and 13 reimbursable').", citations, debug, success=True)

        with _connect() as conn:
            qmarks = ",".join(["?"] * len(ids))
            conn.execute(
                f"UPDATE documents SET reimbursable = 1, updated_at = datetime('now') WHERE doc_id IN ({qmarks})",
                (*ids,),
            )
            conn.commit()

        citations.append("DB:documents")
        resp = (
            f"✅ Marked receipts as reimbursable: {', '.join(map(str, ids))}.\n\n"
            "Checklist of required attachments (typical):\n"
            "- Original receipt image/PDF\n"
            "- Proof of payment if required\n"
            "- Business purpose note (1 line)\n"
        )
        debug["updated_doc_ids"] = ids
        for d in ids:
            citations.append(f"DB:audit_flags.doc_id={d}")
        return _wrap_response(resp, citations, debug, success=True)

    if intent == "explain_flag":
        m = re.search(r"(?:doc|document|receipt)\s*#?(\d+)", user_text, re.IGNORECASE)
        doc_id = int(m.group(1)) if m else None
        if not doc_id:
            with _connect() as conn:
                row = conn.execute("SELECT doc_id FROM audit_flags ORDER BY datetime(created_at) DESC LIMIT 1").fetchone()
            doc_id = int(row[0]) if row else None

        if not doc_id:
            return _wrap_response("No flagged receipts found.", citations, debug, success=True)

        flags = get_audit_flags(doc_id)
        citations.append(f"DB:documents.doc_id={doc_id}")
        citations.append(f"DB:audit_flags.doc_id={doc_id}")

        resp = f"🧾 **Flag Explanation for doc_id={doc_id}**\n\n"
        if not flags:
            resp += "No audit flags recorded.\n"
        else:
            for f in flags[:5]:
                resp += f"• {f['flag_type']}: {f['detail']} (at {f['created_at']})\n"

        resp += "\n**Next steps:** Open the Pending Receipts tab and choose the best candidate values (no free typing)."
        return _wrap_response(resp, citations, debug, success=True)

    if intent == "compare_spending_periods":
        from datetime import datetime
        year = datetime.now().year
        comparison = compare_spending_periods(f"{year}-01-01", f"{year}-01-31", f"{year}-02-01", f"{year}-02-28")
        resp = _format_spending_comparison(comparison, f"{year}-01", f"{year}-02")
        citations.append("DB:documents")
        debug["comparison"] = comparison
        return _wrap_response(resp, citations, debug, success=True)

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
        return _wrap_response(response, citations, debug, success=True)

    if intent == "spending_by_category":
        start_time = time.time()
        debug["routing_model"] = "phi"
        debug["writer_model"] = "formatter"
        
        use_mistral = MODEL_MODE == "phi+mistral" and _LLM_AVAILABLE
        
        if use_mistral:
            try:
                rows = spend_by_category(days=30)
                formatted = _format_category_spending(rows, days=30)
                response = _rewrite_with_mistral(formatted, user_text, "spending_by_category")
                debug["writer_model"] = "mistral"
                debug["models_used"] = ["Phi (router)", "Mistral (writer)"]
            except Exception:
                rows = spend_by_category(days=30)
                response = _format_category_spending(rows, days=30)
                debug["writer_model"] = "formatter"
                debug["models_used"] = ["Phi (router)"]
        else:
            rows = spend_by_category(days=30)
            response = _format_category_spending(rows, days=30)
            debug["models_used"] = ["Phi (router)"]
        
        debug["rows"] = rows if 'rows' in locals() else None
        debug["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        citations.append("DB:documents")
        return _wrap_response(response, citations, debug, success=True)

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
        start_time = time.time()
        debug["routing_model"] = "phi"
        debug["writer_model"] = "formatter"
        
        # Determine which mode to use
        use_chaining = USE_LLM_CHAINING and intent in CHAINABLE_INTENTS and _LLM_AVAILABLE
        use_mistral = MODEL_MODE == "phi+mistral" and _LLM_AVAILABLE and not use_chaining
        
        if use_chaining:
            # Use LLM chaining pipeline
            try:
                tool_schema = "DB Functions: detect_anomalies() returns [{anomaly_type, vendor, details, doc_id}, ...]\nResponse styles: summary, detailed"
                plan = _plan_with_phi(user_text, tool_schema)
                debug["models_used"] = ["Phi (planner)"]
                
                if plan:
                    anomalies = detect_anomalies()
                    db_results = f"Detected anomalies:\n{str(anomalies)}"
                    
                    draft_answer = _write_answer_with_mistral(user_text, db_results)
                    if draft_answer:
                        debug["models_used"].append("Mistral (writer)")
                        debug["writer_model"] = "mistral"
                        
                        verification = _verify_answer_with_phi(draft_answer, db_results)
                        debug["models_used"].append("Phi (verifier)")
                        
                        if verification and verification.get("is_supported"):
                            response = draft_answer
                            if verification.get("revised_answer"):
                                response = verification["revised_answer"]
                        else:
                            response = draft_answer if draft_answer else _format_anomalies(anomalies)
                    else:
                        response = _format_anomalies(anomalies)
                else:
                    anomalies = detect_anomalies()
                    response = _format_anomalies(anomalies)
                
                debug["chain_type"] = "llm_chaining"
            except Exception as e:
                anomalies = detect_anomalies()
                response = _format_anomalies(anomalies)
                debug["chain_error"] = str(e)
                debug["models_used"] = []
        
        elif use_mistral:
            # Use formatter + Mistral rewrite mode
            try:
                anomalies = detect_anomalies()
                formatted = _format_anomalies(anomalies)
                response = _rewrite_with_mistral(formatted, user_text, "anomalies")
                debug["writer_model"] = "mistral"
                debug["models_used"] = ["Phi (router)", "Mistral (writer)"]
            except Exception:
                anomalies = detect_anomalies()
                response = _format_anomalies(anomalies)
                debug["writer_model"] = "formatter"
                debug["models_used"] = ["Phi (router)"]
        
        else:
            # Use deterministic formatter only
            anomalies = detect_anomalies()
            response = _format_anomalies(anomalies)
            debug["models_used"] = ["Phi (router)"]
        
        debug["anomalies"] = anomalies if 'anomalies' in locals() else None
        debug["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        citations.append("DB:documents")
        return _wrap_response(response, citations, debug, success=True)

    if intent == "weekly_summary":
        start_time = time.time()
        debug["routing_model"] = "phi"
        debug["writer_model"] = "formatter"
        
        try:
            rows = spend_by_category_calendar(period="week", n_periods=8)
            
            if not rows:
                response = "No receipt data available for weekly summary."
            else:
                # Format as grouped by week
                weeks_dict = {}
                for period_label, category, total_spend, count in rows:
                    if period_label not in weeks_dict:
                        weeks_dict[period_label] = {}
                    weeks_dict[period_label][category] = {"total": total_spend, "count": count}
                
                response = "**Weekly Spending Summary (Last 8 Weeks)**\n\n"
                for week in sorted(weeks_dict.keys(), reverse=True):
                    response += f"**Week {week}**\n"
                    week_total = 0
                    for category, data in sorted(weeks_dict[week].items()):
                        amount = data["total"]
                        count = data["count"]
                        week_total += amount
                        response += f"  • {category.capitalize()}: ${amount:.2f} ({count} items)\n"
                    response += f"  **Week Total: ${week_total:.2f}**\n\n"
        except Exception:
            response = "Error generating weekly summary."
        
        debug["models_used"] = ["Phi (router)"]
        debug["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        citations.append("DB:documents")
        return _wrap_response(response, citations, debug, success=True)

    if intent == "monthly_summary":
        start_time = time.time()
        debug["routing_model"] = "phi"
        debug["writer_model"] = "formatter"
        
        try:
            rows = spend_by_category_calendar(period="month", n_periods=6)
            
            if not rows:
                response = "No receipt data available for monthly summary."
            else:
                # Format as grouped by month
                months_dict = {}
                for period_label, category, total_spend, count in rows:
                    if period_label not in months_dict:
                        months_dict[period_label] = {}
                    months_dict[period_label][category] = {"total": total_spend, "count": count}
                
                response = "**Monthly Spending Summary (Last 6 Months)**\n\n"
                for month in sorted(months_dict.keys(), reverse=True):
                    response += f"**Month {month}**\n"
                    month_total = 0
                    for category, data in sorted(months_dict[month].items()):
                        amount = data["total"]
                        count = data["count"]
                        month_total += amount
                        response += f"  • {category.capitalize()}: ${amount:.2f} ({count} items)\n"
                    response += f"  **Month Total: ${month_total:.2f}**\n\n"
        except Exception:
            response = "Error generating monthly summary."
        
        debug["models_used"] = ["Phi (router)"]
        debug["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        citations.append("DB:documents")
        return _wrap_response(response, citations, debug, success=True)

    if intent == "pending_receipts":
        start_time = time.time()
        debug["routing_model"] = "phi"
        debug["writer_model"] = "formatter"
        
        try:
            pending = list_pending_receipts(limit=50)
            
            if not pending:
                response = "No pending receipts. All receipts are complete!"
            else:
                # Format as markdown table
                response = f"**Pending Receipts ({len(pending)} total)**\n\n"
                response += "| Doc ID | Vendor | Date | Total | Missing Fields |\n"
                response += "|--------|--------|------|-------|----------------|\n"
                
                for item in pending:
                    doc_id = item.get("doc_id", "")
                    vendor = item.get("vendor") or "(empty)"
                    date = item.get("doc_date") or "(empty)"
                    total = item.get("total") or "0.00"
                    missing = item.get("missing_fields", "none")
                    
                    # Format total as currency if numeric
                    if isinstance(total, (int, float)):
                        total = f"${total:.2f}"
                    
                    response += f"| {doc_id} | {vendor} | {date} | {total} | {missing} |\n"
                
                response += f"\n**Action:** Open the 'Pending Receipts' tab to complete these receipts."
        except Exception:
            response = "Error retrieving pending receipts."
        
        debug["models_used"] = ["Phi (router)"]
        debug["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        citations.append("DB:documents")
        return _wrap_response(response, citations, debug, success=True)

    if intent == "vendor_verification" or "verify" in user_text.lower():
        # Better vendor extraction: keep everything after "verify vendor" / "verify"
        text = user_text.strip()

        vendor_name = None
        for prefix in ["verify vendor", "verify", "official site for", "website for", "check vendor", "check company"]:
            idx = text.lower().find(prefix)
            if idx != -1:
                vendor_name = text[idx + len(prefix):].strip(" :,-")
                break

        # Fallback: try regex (your old one)
        if not vendor_name:
            vendor_match = re.search(r"(?:verify|check)\s+(?:vendor|company)?\s*(.+)", text, re.IGNORECASE)
            vendor_name = vendor_match.group(1).strip() if vendor_match else None

        if not vendor_name or len(vendor_name) < 2:
            response = "Please specify which vendor to verify (e.g., 'Verify vendor Domino\\'s')"
            return _wrap_response(response, citations, debug, success=True)

        result = verify_vendor_online(vendor_name)
        citations.append("WEB:duckduckgo_html")
        debug["vendor_result"] = result

        # Format response
        response = f"🏢 **Vendor Verification: {vendor_name}**\n\n"

        if result.get("error"):
            response += f"**Status:** ✗ Not Verified\n**Error:** {result['error']}\n"
            citations.append("WEB:duckduckgo_html")
            return _wrap_response(response, citations, debug, success=True)

        best = result.get("best_guess_official")
        confidence = result.get("confidence", 0.0)
        response += f"**Confidence:** {confidence*100:.0f}%\n"
        if result.get("note"):
            response += f"**Note:** {result['note']}\n"

        if best:
            response += f"\n✅ **Best guess official site:** {best.get('domain')} — {best.get('url')}\n"
        else:
            response += "\n⚠️ **Best guess official site:** Not confident enough to pick one.\n"

        response += "\n**Top results:**\n"
        for i, r in enumerate(result.get("results", [])[:3], start=1):
            response += f"{i}. {r.get('title')} — {r.get('domain')} — {r.get('url')}\n"

        # Citations: include real URLs for slide credibility
        if best and best.get("url"):
            citations.append(best["url"])
        for r in result.get("results", [])[:3]:
            if r.get("url"):
                citations.append(r["url"])

        return _wrap_response(response, citations, debug, success=True)
    
    # Final fallback: never return None
    response = (
        "I can:\n"
        "- Extract receipts/invoices (OCR) and store them in SQLite\n"
        "- Show recent receipts, spend by vendor/category, duplicates, anomalies\n"
        "- Weekly/monthly summaries and pending receipts editing\n"
        "- Vendor verification and currency conversion\n"
        "- Security protections against prompt injection\n"
    )
    citations = citations if isinstance(citations, list) else []
    return _wrap_response(response, citations, debug, success=True)