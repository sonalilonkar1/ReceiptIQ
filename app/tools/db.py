"""Database helper utilities."""

from pathlib import Path
import sqlite3


DB_PATH = Path(__file__).resolve().parents[2] / "receiptiq.sqlite"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def insert_document(extracted: dict) -> int:
    """Insert a parsed document and return its generated doc_id."""
    values = (
        extracted.get("doc_type"),
        extracted.get("vendor"),
        extracted.get("doc_date"),
        extracted.get("currency"),
        extracted.get("subtotal"),
        extracted.get("tax"),
        extracted.get("total"),
        extracted.get("confidence"),
        extracted.get("raw_text"),
    )

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents (
                doc_type,
                vendor,
                doc_date,
                currency,
                subtotal,
                tax,
                total,
                confidence,
                raw_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        conn.commit()
        return int(cursor.lastrowid)


def add_flag(doc_id: int, flag_type: str, detail: str) -> None:
    """Insert an audit flag row for a document."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_flags (doc_id, flag_type, detail)
            VALUES (?, ?, ?)
            """,
            (doc_id, flag_type, detail),
        )
        conn.commit()


def get_recent_docs(limit: int = 10) -> list[tuple]:
    """Return recent documents as tuples.

    Shape: (doc_id, vendor, doc_date, currency, total, created_at)
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT doc_id, vendor, doc_date, currency, total, created_at
            FROM documents
            ORDER BY datetime(created_at) DESC, doc_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()


def spend_by_vendor() -> list[tuple]:
    """Return spend totals grouped by vendor.

    Shape: (vendor, sum_total)
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT vendor, COALESCE(SUM(total), 0) AS sum_total
            FROM documents
            GROUP BY vendor
            ORDER BY sum_total DESC
            """
        )
        return cursor.fetchall()


def find_duplicates() -> list[tuple]:
    """Return potential duplicate documents.

    Shape: (vendor, doc_date, total, count)
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT vendor, doc_date, total, COUNT(*) AS count
            FROM documents
            GROUP BY vendor, doc_date, total
            HAVING COUNT(*) > 1
            ORDER BY count DESC, vendor, doc_date
            """
        )
        return cursor.fetchall()
