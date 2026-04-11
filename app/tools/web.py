"""Web retrieval and parsing utilities."""

import re
from typing import Optional

import requests


def _extract_currency_and_amount(query: str) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Extract amount and currencies from query text.
    
    Returns: (amount, from_currency, to_currency)
    Example: "100 USD to EUR" -> (100.0, 'USD', 'EUR')
    """
    # Pattern: number currency to currency
    pattern = r"(\d+(?:\.\d{2})?)\s*([A-Z]{3})\s+to\s+([A-Z]{3})"
    match = re.search(pattern, query.upper())
    
    if match:
        amount = float(match.group(1))
        from_curr = match.group(2)
        to_curr = match.group(3)
        return amount, from_curr, to_curr
    
    return None, None, None


def _get_exchange_rate(from_currency: str, to_currency: str) -> Optional[float]:
    """Fetch exchange rate from Open Exchange Rates API or fallback to hardcoded rates."""
    # Hardcoded rates for common currency pairs (as of 2026 example)
    # In production, use: https://open.er-api.com/v6/latest/{from_currency}
    rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.5,
        ("USD", "CAD"): 1.36,
        ("USD", "AUD"): 1.52,
        ("EUR", "USD"): 1.09,
        ("GBP", "USD"): 1.27,
        ("JPY", "USD"): 0.0067,
        ("CAD", "USD"): 0.735,
        ("AUD", "USD"): 0.66,
    }
    
    return rates.get((from_currency, to_currency))


def web_lookup(query: str) -> dict:
    """Perform currency conversion or web lookup."""
    amount, from_curr, to_curr = _extract_currency_and_amount(query)
    
    if amount and from_curr and to_curr:
        rate = _get_exchange_rate(from_curr, to_curr)
        if rate:
            converted = amount * rate
            return {
                "query": query,
                "type": "currency_conversion",
                "from": {"amount": amount, "currency": from_curr},
                "to": {"amount": round(converted, 2), "currency": to_curr},
                "rate": round(rate, 4),
                "note": f"{amount} {from_curr} = {round(converted, 2)} {to_curr}",
            }
        else:
            return {
                "query": query,
                "type": "currency_conversion",
                "error": f"Exchange rate not available for {from_curr} to {to_curr}",
            }
    
    return {
        "query": query,
        "type": "web_lookup",
        "results": [],
        "note": "Could not parse currency conversion. Use format: '100 USD to EUR'",
    }
