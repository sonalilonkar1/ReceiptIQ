"""Web retrieval and parsing utilities.

Provides:
- Currency conversion with hardcoded rates
- Vendor verification via DuckDuckGo HTML search with in-memory caching (24h TTL)
"""

import re
import time
from typing import Optional
from urllib.parse import quote
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup


# ============================================================================
# CURRENCY CONVERSION (existing functionality)
# ============================================================================

def _extract_currency_and_amount(query: str) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Extract amount and currencies from query text.
    
    Returns: (amount, from_currency, to_currency)
    Example: "100 USD to EUR" -> (100.0, 'USD', 'EUR')
    """
    # Pattern: number currency to currency (case-insensitive)
    pattern = r"(\d+(?:\.\d{2})?)\s*([A-Z]{3})\s+TO\s+([A-Z]{3})"
    match = re.search(pattern, query.upper())
    
    if match:
        amount = float(match.group(1))
        from_curr = match.group(2)
        to_curr = match.group(3)
        return amount, from_curr, to_curr
    
    return None, None, None


def _get_exchange_rate(from_currency: str, to_currency: str) -> Optional[float]:
    """Fetch exchange rate from hardcoded rates (fallback).
    
    In production, use: https://open.er-api.com/v6/latest/{from_currency}
    """
    # Hardcoded rates for common currency pairs (as of 2026 example)
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


# ============================================================================
# VENDOR VERIFICATION (NEW)
# ============================================================================

# In-memory cache: {vendor_name: (timestamp, result_dict)}
_VENDOR_CACHE = {}
_CACHE_TTL_HOURS = 24


def _is_cache_valid(cached_time: float, ttl_hours: int = _CACHE_TTL_HOURS) -> bool:
    """Check if cached result is still valid (within TTL)."""
    return (time.time() - cached_time) < (ttl_hours * 3600)


def _clean_vendor_name(vendor: str) -> str:
    """Clean vendor name to tokens for matching.
    
    Example: "Starbucks Coffee Co." -> ["starbucks", "coffee"]
    """
    # Remove special chars, lowercase, split
    cleaned = re.sub(r'[^a-z0-9\s]', '', vendor.lower())
    return cleaned.split()


def _extract_domain(url: str) -> str:
    """Extract domain from URL.
    
    Example: "https://www.example.com/path" -> "example.com"
    """
    try:
        # Remove protocol
        domain = url.replace("https://", "").replace("http://", "")
        # Get domain part (remove path)
        domain = domain.split("/")[0]
        # Remove www prefix
        domain = domain.replace("www.", "")
        return domain
    except Exception:
        return ""


def _is_suspicious_domain(domain: str) -> bool:
    """Check if domain is a known aggregator/social site (penalize these).
    
    Penalize: facebook, yelp, maps, linkedin, instagram, wikipedia, tripadvisor, etc.
    """
    suspicious = {
        "facebook", "yelp", "maps", "google.com/maps", "linkedin", 
        "instagram", "wikipedia", "tripadvisor", "twitter", "pinterest",
        "reddit", "amazon", "ebay", "yellowpages", "foursquare"
    }
    return any(sus in domain.lower() for sus in suspicious)


def _score_result(title: str, url: str, vendor_tokens: list[str]) -> float:
    """Score a search result for how likely it is the official vendor site.
    
    Returns score 0.0-1.0 based on heuristics.
    """
    score = 0.5  # Base score
    
    # Bonus for https
    if url.startswith("https://"):
        score += 0.15
    
    # Bonus for matching vendor name tokens in title
    title_lower = title.lower()
    matches = sum(1 for token in vendor_tokens if token in title_lower)
    if matches > 0:
        score += min(0.25, matches * 0.1)
    
    # Bonus for matching vendor name in domain
    domain = _extract_domain(url)
    domain_lower = domain.lower()
    domain_matches = sum(1 for token in vendor_tokens if token in domain_lower)
    if domain_matches > 0:
        score += min(0.35, domain_matches * 0.15)
    
    # Penalty for suspicious domain
    if _is_suspicious_domain(domain):
        score -= 0.4
    
    # Ensure score stays in [0, 1]
    return max(0.0, min(1.0, score))


def _parse_duckduckgo_results(html: str, max_results: int = 3) -> list[dict]:
    """Parse DuckDuckGo HTML results.
    
    Returns list of dicts with: {title, url, snippet, domain}
    """
    results = []
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # DuckDuckGo HTML result structure: <div class="result"> with <a class="result__a"> for URL
        result_divs = soup.find_all('div', class_='result')
        
        for result_div in result_divs[:max_results]:
            try:
                # Extract title and URL
                link = result_div.find('a', class_='result__a')
                if not link or not link.get('href'):
                    continue
                
                title = link.get_text(strip=True)
                url = link.get('href', '')
                
                # Extract snippet
                snippet_elem = result_div.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Extract domain
                domain = _extract_domain(url)
                
                if title and url:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "domain": domain,
                    })
            except Exception:
                continue
        
        return results
    
    except Exception as e:
        return []


def verify_vendor_online(vendor_name: str, timeout_s: float = 2.0) -> dict:
    """Verify vendor online using DuckDuckGo HTML search.
    
    Args:
        vendor_name: Name of vendor to verify (e.g., "Starbucks")
    
    Returns:
        dict with:
        - type: "vendor_verification"
        - vendor: vendor_name
        - best_guess_official: {title, url, domain} or None
        - results: list of top 3 results
        - confidence: float 0-1
        - source: "duckduckgo_html"
        - note: explanation
        - cached: bool (whether result came from cache)
    """
    if not vendor_name or not vendor_name.strip():
        return {
            "type": "vendor_verification",
            "vendor": vendor_name,
            "error": "Vendor name cannot be empty",
            "results": [],
            "best_guess_official": None,
            "confidence": 0.0,
            "source": "duckduckgo_html",
        }
    
    vendor_lower = vendor_name.lower().strip()
    
    # Check cache
    if vendor_lower in _VENDOR_CACHE:
        cached_time, cached_result = _VENDOR_CACHE[vendor_lower]
        if _is_cache_valid(cached_time):
            cached_result["cached"] = True
            return cached_result
        else:
            # Cache expired, remove it
            del _VENDOR_CACHE[vendor_lower]
    
    # Perform search
    try:
        query = f"{vendor_name} official website"
        search_url = f"https://duckduckgo.com/html/?q={quote(query)}"
        
        # Fetch with timeout and User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(search_url, headers=headers, timeout=timeout_s)
        response.raise_for_status()
        
        # Parse results
        results = _parse_duckduckgo_results(response.text, max_results=3)
        
        if not results:
            result = {
                "type": "vendor_verification",
                "vendor": vendor_name,
                "best_guess_official": None,
                "results": [],
                "confidence": 0.0,
                "source": "duckduckgo_html",
                "note": "No search results found",
                "cached": False,
            }
        else:
            # Score results for best guess
            vendor_tokens = _clean_vendor_name(vendor_name)
            scored_results = [
                (r, _score_result(r["title"], r["url"], vendor_tokens))
                for r in results
            ]
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            best_result = scored_results[0]
            best_score = best_result[1]
            best_guess = best_result[0] if best_score > 0.4 else None
            
            result = {
                "type": "vendor_verification",
                "vendor": vendor_name,
                "best_guess_official": best_guess,
                "results": results,
                "confidence": round(best_score, 2),
                "source": "duckduckgo_html",
                "note": (
                    f"Best guess: {best_guess['domain'] if best_guess else 'None'} "
                    f"(confidence: {round(best_score * 100)}%)"
                ),
                "cached": False,
            }
        
        # Cache the result
        _VENDOR_CACHE[vendor_lower] = (time.time(), result)
        
        return result
    
    except requests.exceptions.Timeout:
        return {
            "type": "vendor_verification",
            "vendor": vendor_name,
            "error": "Request timeout (web search unavailable)",
            "results": [],
            "best_guess_official": None,
            "confidence": 0.0,
            "source": "duckduckgo_html",
            "cached": False,
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "type": "vendor_verification",
            "vendor": vendor_name,
            "error": f"Web search failed: {str(e)[:50]}",
            "results": [],
            "best_guess_official": None,
            "confidence": 0.0,
            "source": "duckduckgo_html",
            "cached": False,
        }
    
    except Exception as e:
        return {
            "type": "vendor_verification",
            "vendor": vendor_name,
            "error": f"Unexpected error: {str(e)[:50]}",
            "results": [],
            "best_guess_official": None,
            "confidence": 0.0,
            "source": "duckduckgo_html",
            "cached": False,
        }


def clear_vendor_cache() -> int:
    """Clear vendor verification cache. Returns number of entries cleared."""
    global _VENDOR_CACHE
    count = len(_VENDOR_CACHE)
    _VENDOR_CACHE.clear()
    return count


def normalize_vendor_name(vendor: str) -> str:
    """Normalize vendor to a cleaner canonical form (basic cleanup)."""
    if not vendor:
        return ""
    v = vendor.strip()
    v = re.sub(r"\s+", " ", v)
    v = re.sub(r"[^A-Za-z0-9&.'\- ]+", "", v).strip()
    return v
