"""LLM-based receipt text parser using Ollama (local models: Phi, Mistral)."""

from __future__ import annotations

import json
import re
from typing import Optional
import requests

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 60  # seconds


def _normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Normalize date to YYYY-MM-DD format."""
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Try YYYY-MM-DD (already normalized)
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
    # Try MM/DD/YYYY or M/D/YYYY
    match = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', date_str)
    if match:
        month, day, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Try YYYY/MM/DD
    match = re.match(r'^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    return None


def _check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def parse_receipt_text_with_ollama(raw_text: str, model: str = "phi") -> dict:
    """Parse receipt OCR text using Ollama to extract structured fields.
    
    Args:
        raw_text: OCR-extracted receipt text
        model: "phi" (default, faster, 128M) or "mistral" (slower, 7B)
    
    Returns:
        Dict with fields: vendor, date, total, subtotal, tax, currency, category,
        invoice_number, line_items (optional).
        
        Each field has structure:
        {
            "vendor": {"value": str|null, "confidence": 0-1, "evidence_line": str|null},
            "date": {"value": str|null, "confidence": 0-1, "evidence_line": str|null},
            ...
        }
    """
    # Check if Ollama is running
    if not _check_ollama_available():
        return _empty_result("Ollama not available. Run: ollama serve")
    
    try:
        # Validate model is available
        valid_models = {"phi", "mistral"}
        if model.lower() not in valid_models:
            return _empty_result(f"Model '{model}' not available. Try: {valid_models}")
        
        # Truncate if too long (keep last 2000 chars where amounts usually are)
        text_to_parse = raw_text[-2000:] if len(raw_text) > 2000 else raw_text
        
        extraction_prompt = f"""You are a receipt parser. Extract fields ONLY if explicitly present in the receipt text.

Receipt text:
{text_to_parse}

Return ONLY this JSON structure (use null for missing fields, all amounts as numbers):
{{
  "vendor": {{"value": "name or null", "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}},
  "date": {{"value": "YYYY-MM-DD or null", "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}},
  "total": {{"value": number or null, "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}},
  "subtotal": {{"value": number or null, "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}},
  "tax": {{"value": number or null, "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}},
  "currency": {{"value": "USD or other", "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}},
  "category": {{"value": "meals|travel|supplies|other", "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}},
  "invoice_number": {{"value": "number or null", "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}},
  "line_items": {{"value": ["item1", "item2"] or [], "confidence": 0.0-1.0, "evidence_line": "exact line from text or null"}}
}}

Rules:
1. ONLY extract values explicitly present in the text
2. confidence: 1.0 if very clear, 0.8 if probable, 0.5 if uncertain, 0.0 if missing
3. evidence_line: exact text from receipt supporting the extraction
4. Dates must be normalized to YYYY-MM-DD
5. Amounts must be numbers (remove $, commas)
6. Return ONLY JSON, no other text

JSON:"""
        
        # Call Ollama API
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model.lower(),
                "prompt": extraction_prompt,
                "stream": False,
                "temperature": 0.3,
            },
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code != 200:
            return _empty_result(f"Ollama error: {response.status_code}")
        
        response_data = response.json()
        response_text = response_data.get("response", "")
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            
            try:
                parsed = json.loads(json_str)
                result = _validate_and_normalize_result(parsed)
                return result
            except json.JSONDecodeError as e:
                return _empty_result(f"JSON parse error: {e}")
        else:
            return _empty_result("No JSON found in Ollama response")
    
    except requests.exceptions.ConnectionError:
        return _empty_result("Cannot connect to Ollama. Run: ollama serve")
    except requests.exceptions.Timeout:
        return _empty_result("Ollama request timed out")
    except Exception as e:
        return _empty_result(f"Ollama parsing error: {str(e)}")


def _empty_result(reason: str = "") -> dict:
    """Return empty result structure with all fields null."""
    return {
        "vendor": {"value": None, "confidence": 0.0, "evidence_line": None},
        "date": {"value": None, "confidence": 0.0, "evidence_line": None},
        "total": {"value": None, "confidence": 0.0, "evidence_line": None},
        "subtotal": {"value": None, "confidence": 0.0, "evidence_line": None},
        "tax": {"value": None, "confidence": 0.0, "evidence_line": None},
        "currency": {"value": "USD", "confidence": 0.5, "evidence_line": None},
        "category": {"value": None, "confidence": 0.0, "evidence_line": None},
        "invoice_number": {"value": None, "confidence": 0.0, "evidence_line": None},
        "line_items": {"value": [], "confidence": 0.0, "evidence_line": None},
        "error": reason
    }


def _validate_and_normalize_result(parsed: dict) -> dict:
    """Validate and normalize Ollama response."""
    result = _empty_result()
    
    def process_field(field_name: str, normalize_fn=None):
        """Process a single field from parsed response."""
        if field_name not in parsed:
            return
        
        field_data = parsed[field_name]
        if isinstance(field_data, dict):
            value = field_data.get("value")
            confidence = field_data.get("confidence", 0.0)
            evidence_line = field_data.get("evidence_line")
        else:
            # Handle raw value (not in dict format)
            value = field_data
            confidence = 0.8 if field_data else 0.0
            evidence_line = None
        
        # Normalize value if function provided
        if normalize_fn and value is not None:
            value = normalize_fn(value)
        
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, float(confidence) if confidence else 0.0))
        
        result[field_name] = {
            "value": value,
            "confidence": confidence,
            "evidence_line": evidence_line
        }
    
    # Process fields with appropriate normalization
    process_field("vendor", lambda x: str(x).strip() if x else None)
    process_field("date", _normalize_date)
    process_field("total", lambda x: float(x) if x else None)
    process_field("subtotal", lambda x: float(x) if x else None)
    process_field("tax", lambda x: float(x) if x else None)
    process_field("currency", lambda x: str(x).upper() if x else "USD")
    process_field("category", lambda x: str(x).lower() if x else None)
    process_field("invoice_number", lambda x: str(x).strip() if x else None)
    
    # Handle line_items (can be list or string)
    if "line_items" in parsed:
        items_data = parsed["line_items"]
        if isinstance(items_data, dict):
            items_value = items_data.get("value", [])
            items_confidence = items_data.get("confidence", 0.0)
        else:
            items_value = items_data
            items_confidence = 0.8 if items_data else 0.0
        
        # Ensure it's a list
        if isinstance(items_value, str):
            items_value = [items_value]
        elif not isinstance(items_value, list):
            items_value = []
        
        result["line_items"] = {
            "value": items_value[:10],  # Limit to 10 items
            "confidence": max(0.0, min(1.0, float(items_confidence))),
            "evidence_line": None
        }
    
    return result


# Backward compatibility: alias for parse_receipt_text_with_llm
parse_receipt_text_with_llm = parse_receipt_text_with_ollama
