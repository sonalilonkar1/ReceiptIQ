"""Web retrieval and parsing utilities."""


def web_lookup(query: str) -> dict:
    """Stub web lookup response for MVP integration."""
    # TODO: Add currency conversion enrichment using trusted FX APIs.
    # TODO: Add vendor verification against web/business registry sources.
    return {
        "query": query,
        "results": [],
        "note": "Web retrieval not implemented yet (stub).",
    }
