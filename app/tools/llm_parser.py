"""ReceiptIQ LLM Parser (Ollama) - Reliable on Mac (Phi)

Problem observed:
- Phi sometimes refuses with: "cannot return only one JSON object..." because it
  misreads the prompt as requiring JSON INPUT.

Fixes:
- Prompt explicitly states the INPUT is plain receipt text (not JSON).
- Provide clear examples and a "you MUST comply" instruction.
- Add a 'system' message to Ollama request to reduce refusals.
- If the model still refuses or returns non-JSON, fall back to regex extraction
  from the input text (grounded) for date/amounts so pipeline doesn't break.

Env vars:
  RECEIPTIQ_LLM_MODEL (default: phi)
  RECEIPTIQ_OLLAMA_TIMEOUT (default: 120 seconds - Phi can be slow on first requests)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests

OLLAMA_API_URL = os.environ.get("RECEIPTIQ_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.environ.get("RECEIPTIQ_OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")

DEFAULT_MODEL = os.environ.get("RECEIPTIQ_LLM_MODEL", "phi")
FALLBACK_MODEL = "phi"

OLLAMA_TIMEOUT = int(os.environ.get("RECEIPTIQ_OLLAMA_TIMEOUT", "120"))  # 2 minutes - Phi needs time on first requests
MAX_RETRIES = 1

_SESSION = requests.Session()
_WARMED_MODELS: set[str] = set()


def _empty_result(error: str | None = None) -> Dict[str, Dict[str, Any]]:
    if error:
        print(f">>> DEBUG: LLM PARSER EMPTY RESULT: {error}", file=sys.stderr)
        sys.stderr.flush()
    return {
        "vendor": {"value": None, "confidence": 0.0, "evidence_line": None},
        "date": {"value": None, "confidence": 0.0, "evidence_line": None},
        "total": {"value": None, "confidence": 0.0, "evidence_line": None},
        "subtotal": {"value": None, "confidence": 0.0, "evidence_line": None},
        "tax": {"value": None, "confidence": 0.0, "evidence_line": None},
    }


def _check_ollama_available() -> bool:
    try:
        r = _SESSION.get(OLLAMA_TAGS_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _warmup_model(model: str) -> None:
    if model in _WARMED_MODELS:
        return
    if not _check_ollama_available():
        return
    try:
        _SESSION.post(
            OLLAMA_API_URL,
            json={"model": model, "prompt": "hi", "stream": False, "options": {"temperature": 0.0, "num_predict": 8}},
            timeout=(3, min(15, OLLAMA_TIMEOUT)),
        )
        _WARMED_MODELS.add(model)
    except Exception:
        return


def _normalize_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    s = str(date_str).strip()
    s = s.split(" ")[0].strip()
    s = s.replace(".", "/").replace("\\", "/")

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s

    fmts = [
        "%d-%m-%y", "%d/%m/%y",
        "%d-%m-%Y", "%d/%m/%Y",
        "%m/%d/%y", "%m-%d-%y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%Y-%m-%d",
    ]

    cand = [s, s.replace("/", "-")]
    for t in cand:
        for fmt in fmts:
            try:
                dt = datetime.strptime(t, fmt)
                if dt.year < 1970:
                    dt = dt.replace(year=dt.year + 100)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue

    if re.fullmatch(r"\d{8}", s):
        try:
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return None
    return None


def _extract_date_from_text(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", raw_text)
    if m:
        return _normalize_date(m.group(1))
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw_text)
    if m:
        return _normalize_date(m.group(1))
    return None


def _parse_amount(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", "")
    s = re.sub(r"(?i)\b(rm|usd|eur|gbp|inr|cad|aud|jpy)\b", "", s)
    s = s.replace("$", "").replace("€", "").replace("£", "")
    m = re.search(r"(\d+\.\d{2})", s)
    if not m:
        m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _regex_amounts_from_text(raw_text: str) -> Dict[str, Any]:
    """Very small fallback for amounts when LLM fails."""
    out: Dict[str, Any] = {}
    t = raw_text or ""
    # Total: prefer lines containing TOTAL
    m = re.search(r"(?i)\btotal\b[^0-9]{0,20}(\d+[.,]\d{2})", t)
    if m:
        out["total"] = m.group(1)
    # Subtotal
    m = re.search(r"(?i)\bsub\s*total\b[^0-9]{0,20}(\d+[.,]\d{2})", t)
    if m:
        out["subtotal"] = m.group(1)
    # Tax
    m = re.search(r"(?i)\b(tax|gst|vat)\b[^0-9]{0,20}(\d+[.,]\d{2})", t)
    if m:
        out["tax"] = m.group(2)
    return out

def _extract_vendor_from_text(raw_text: str) -> Optional[str]:
    """
    Lightweight vendor fallback when LLM fails.
    Heuristic: choose first non-empty line with enough letters, not an address/metadata.
    """
    if not raw_text:
        return None
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    bad = re.compile(r"(?i)\b(total|subtotal|tax|gst|vat|amount|change|cashier|server|table|thank|welcome|invoice|receipt)\b")
    addr = re.compile(r"(?i)\b(st|street|rd|road|ave|avenue|blvd|lane|ln|dr|drive|zip|postcode|tel|phone)\b")
    for ln in lines[:12]:
        clean = re.sub(r"[^A-Za-z0-9&.'/- ]+", "", ln).strip()
        if len(clean) < 3:
            continue
        # reject if looks like address or metadata
        if bad.search(clean) or addr.search(clean):
            continue
        letters = sum(ch.isalpha() for ch in clean)
        if letters < 3:
            continue
        # reject if mostly digits
        if sum(ch.isdigit() for ch in clean) / max(1, len(clean)) > 0.2:
            continue
        return clean
    return None


def _find_evidence_line(raw_text: str, value: Any) -> Optional[str]:
    """Return the OCR line that most directly supports a predicted value (best-effort)."""
    if not raw_text or value in (None, ""):
        return None
    val = str(value).strip().lower()
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    # For totals/dates, search by numeric substring; for vendor search by normalized substring
    if re.search(r"\d", val):
        token = re.sub(r"[^0-9./-]+", "", val)
        if token:
            for ln in lines:
                if token in ln.lower():
                    return ln[:200]
    norm = re.sub(r"[^a-z0-9]+", " ", val).strip()
    for ln in lines:
        lnorm = re.sub(r"[^a-z0-9]+", " ", ln.lower()).strip()
        if norm and norm in lnorm:
            return ln[:200]
    return None


def _extract_balanced_json(text: str) -> Optional[str]:
    if not text:
        return None
    s = str(text)
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE)

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _jsonish_to_dict(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    j = _extract_balanced_json(text)
    if j:
        j2 = re.sub(r"//.*?$", "", j, flags=re.MULTILINE)
        j2 = re.sub(r",\s*([}\]])", r"\1", j2)
        try:
            obj = json.loads(j2)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    # fallback: key/value lines
    out: Dict[str, Any] = {}
    for line in str(text).splitlines():
        line = line.strip().strip(",")
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().strip('"').lower()
        v = v.strip().strip('"').strip()
        if k in {"date", "total", "subtotal", "tax"}:
            out[k] = v
    return out


def _looks_like_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in [
        "i cannot", "i can't", "cannot return", "please provide", "required format", "not in the required format"
    ])


def _prompt(text_to_parse: str) -> str:
    # Explicitly say input is NOT JSON.
    # We ask for vendor + date + totals in a single JSON object.
    return f"""
You are an information extraction function.

INPUT:
- The input below is plain text from an OCR'd receipt. It is NOT JSON.

TASK:
- Extract these fields:
  1) vendor (store/merchant/company name)
  2) date (transaction date)
  3) total (final amount paid)
  4) subtotal (optional)
  5) tax (optional)
- Output a SINGLE JSON object ONLY (no markdown, no extra words).
- Use null if a field is missing.
- date must be YYYY-MM-DD (normalize 2-digit years like 18 -> 2018).
- amounts must be numbers (no currency symbols).

RULES / HINTS:
- Vendor is usually near the top and is mostly letters (not an address, not a slogan, not 'TOTAL').
- Total is usually the final payable amount (often near lines with TOTAL / AMOUNT DUE / GRAND TOTAL).
- If multiple totals appear, choose the one most likely to be the final total (often the largest near 'TOTAL').

Example:
Input text:
99 SPEED MART S/B
Date: 13/05/2018
TOTAL 22.90

Output:
{{"vendor":"99 SPEED MART S/B","date":"2018-05-13","total":22.90,"subtotal":null,"tax":null}}

Now extract from this receipt text:
{text_to_parse}

Output JSON:
""".strip()



def _call_ollama(model: str, prompt: str) -> Tuple[str, float]:
    t0 = time.perf_counter()
    try:
        resp = _SESSION.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                # system helps some models comply
                "system": "You output JSON only. Do not refuse. The input is plain text, not JSON.",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 80,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["\n\n", "```"],
                },
            },
            timeout=(3, OLLAMA_TIMEOUT),  # 3s connect, 60s read
        )
        ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return str(data.get("response", "")), ms
    except requests.Timeout as e:
        ms = (time.perf_counter() - t0) * 1000.0
        print(f">>> TIMEOUT calling {model} after {ms:.0f}ms: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        print(f">>> ERROR calling {model} after {ms:.0f}ms: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise


def parse_receipt_text_with_llm(raw_text: str, model: str = DEFAULT_MODEL) -> Dict[str, Dict[str, Any]]:
    if not raw_text or not raw_text.strip():
        return _empty_result("empty OCR text")
    if not _check_ollama_available():
        return _empty_result("ollama not available (is `ollama serve` running?)")

    _warmup_model(model)
    if model != FALLBACK_MODEL:
        _warmup_model(FALLBACK_MODEL)

    text_to_parse = raw_text.strip()
    if len(text_to_parse) > 2500:
        text_to_parse = text_to_parse[:2500]

    date_from_text = _extract_date_from_text(text_to_parse)
    prompt = _prompt(text_to_parse)

    attempts = [model] + ([FALLBACK_MODEL] if model != FALLBACK_MODEL else [])
    last_err: Optional[str] = None

    for attempt_model in attempts:
        for retry in range(MAX_RETRIES + 1):
            try:
                response_text, latency_ms = _call_ollama(attempt_model, prompt)

                if _looks_like_refusal(response_text):
                    # Treat as failure but allow fallback logic
                    last_err = f"{attempt_model} refusal"
                    obj = {}
                else:
                    obj = _jsonish_to_dict(response_text)

                # If no JSON, try regex amounts fallback (still grounded)
                if not obj:
                    obj = _regex_amounts_from_text(text_to_parse)

                if not obj:
                    excerpt = response_text[:200].replace("\n", " ")
                    last_err = f"{attempt_model} returned non-JSON (excerpt: {excerpt})"
                    break

                result = _empty_result(None)
                v = obj.get("vendor") or obj.get("company") or obj.get("merchant") or obj.get("store_name")
                if not v:
                    v = _extract_vendor_from_text(text_to_parse)
                if v:
                    result["vendor"]["value"] = str(v).strip()
                    result["vendor"]["confidence"] = 0.80
                    result["vendor"]["evidence_line"] = _find_evidence_line(text_to_parse, v)

                d = date_from_text or (_normalize_date(obj.get("date")) if obj.get("date") else None)
                if d:
                    result["date"]["value"] = d
                    result["date"]["confidence"] = 0.95 if date_from_text else 0.80
                    result["date"]["evidence_line"] = _find_evidence_line(text_to_parse, d)

                total = _parse_amount(obj.get("total"))
                subtotal = _parse_amount(obj.get("subtotal"))
                tax = _parse_amount(obj.get("tax"))

                if total is not None:
                    result["total"]["value"] = round(total, 2)
                    result["total"]["confidence"] = 0.85
                    result["total"]["evidence_line"] = _find_evidence_line(text_to_parse, total)
                if subtotal is not None:
                    result["subtotal"]["value"] = round(subtotal, 2)
                    result["subtotal"]["confidence"] = 0.80
                    result["subtotal"]["evidence_line"] = _find_evidence_line(text_to_parse, subtotal)
                if tax is not None:
                    result["tax"]["value"] = round(tax, 2)
                    result["tax"]["confidence"] = 0.80
                    result["tax"]["evidence_line"] = _find_evidence_line(text_to_parse, tax)

                print(
                    f">>> DEBUG: {attempt_model} parsed in {latency_ms:.0f}ms -> "
                    f"date={result['date']['value']} total={result['total']['value']}",
                    file=sys.stderr,
                )
                sys.stderr.flush()
                return result

            except requests.Timeout:
                last_err = f"{attempt_model} timeout after {OLLAMA_TIMEOUT}s"
                if retry < MAX_RETRIES:
                    time.sleep(1.0)
                    continue
                break
            except Exception as e:
                last_err = f"{attempt_model} error: {e}"
                break

    return _empty_result(last_err or "unknown error")
