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
        extracted.get("category"),
        extracted.get("line_items"),
        extracted.get("description"),
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
                category,
                line_items,
                description,
                raw_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def get_document_by_id(doc_id: int) -> dict:
    """Retrieve a full document by ID.
    
    Returns: dict with all document fields
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT doc_id, doc_type, vendor, doc_date, currency, subtotal, tax, total, confidence, 
                   category, line_items, description, invoice_number, reimbursable, raw_text, created_at
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {}
        
        return {
            "doc_id": row[0],
            "doc_type": row[1],
            "vendor": row[2],
            "doc_date": row[3],
            "currency": row[4],
            "subtotal": row[5],
            "tax": row[6],
            "total": row[7],
            "confidence": row[8],
            "category": row[9],
            "line_items": row[10],
            "description": row[11],
            "invoice_number": row[12],
            "reimbursable": row[13],
            "raw_text": row[14],
            "created_at": row[15],
        }


def get_audit_flags(doc_id: int) -> list[dict]:
    """Get all audit flags for a document."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT flag_id, flag_type, detail, created_at
            FROM audit_flags
            WHERE doc_id = ?
            ORDER BY created_at DESC
            """,
            (doc_id,),
        )
        return [
            {
                "flag_id": row[0],
                "flag_type": row[1],
                "detail": row[2],
                "created_at": row[3],
            }
            for row in cursor.fetchall()
        ]


def spend_by_category(days: int = 30) -> list[tuple]:
    """Return spend totals grouped by category for last N days.
    
    Shape: (category, sum_total, count)
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT category, COALESCE(SUM(total), 0) AS sum_total, COUNT(*) AS count
            FROM documents
            WHERE datetime(created_at) >= datetime('now', '-' || ? || ' days')
            GROUP BY category
            ORDER BY sum_total DESC
            """,
            (days,),
        )
        return cursor.fetchall()


def find_missing_fields() -> list[dict]:
    """Return documents with missing required fields."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT doc_id, vendor, doc_date, category, total, 
                   CASE 
                       WHEN vendor IS NULL THEN 'vendor'
                       WHEN doc_date IS NULL THEN 'date'
                       WHEN total IS NULL THEN 'total'
                       WHEN category IS NULL THEN 'category'
                       ELSE NULL
                   END as missing_field
            FROM documents
            WHERE vendor IS NULL OR doc_date IS NULL OR total IS NULL OR category IS NULL
            ORDER BY created_at DESC
            """
        )
        return [
            {
                "doc_id": row[0],
                "vendor": row[1],
                "doc_date": row[2],
                "category": row[3],
                "total": row[4],
                "missing_field": row[5],
            }
            for row in cursor.fetchall()
        ]


def find_by_amount_threshold(min_amount: float = 0, max_amount: float = None, days: int = 90) -> list[tuple]:
    """Find receipts by amount threshold within date range.
    
    Shape: (doc_id, vendor, total, doc_date, created_at)
    """
    with _connect() as conn:
        if max_amount:
            cursor = conn.execute(
                """
                SELECT doc_id, vendor, total, doc_date, created_at
                FROM documents
                WHERE total >= ? AND total <= ? 
                  AND datetime(created_at) >= datetime('now', '-' || ? || ' days')
                ORDER BY total DESC, created_at DESC
                """,
                (min_amount, max_amount, days),
            )
        else:
            cursor = conn.execute(
                """
                SELECT doc_id, vendor, total, doc_date, created_at
                FROM documents
                WHERE total >= ? 
                  AND datetime(created_at) >= datetime('now', '-' || ? || ' days')
                ORDER BY total DESC, created_at DESC
                """,
                (min_amount, days),
            )
        return cursor.fetchall()


def average_spend_per_period(period: str = "week") -> float:
    """Calculate average spend per week/month.
    
    Period: 'week' or 'month'
    Handles NULL doc_date by using created_at.
    """
    with _connect() as conn:
        if period.lower() == "week":
            cursor = conn.execute(
                """
                SELECT AVG(weekly_sum) FROM (
                    SELECT SUM(total) as weekly_sum
                    FROM documents
                    GROUP BY strftime('%Y-W%W', COALESCE(doc_date, substr(created_at, 1, 10)))
                )
                """
            )
        else:
            cursor = conn.execute(
                """
                SELECT AVG(monthly_sum) FROM (
                    SELECT SUM(total) as monthly_sum
                    FROM documents
                    GROUP BY strftime('%Y-%m', COALESCE(doc_date, substr(created_at, 1, 10)))
                )
                """
            )
        result = cursor.fetchone()
        return float(result[0]) if result[0] else 0.0


def check_expense_rules_violations(rule_name: str = "lunch_limit") -> list[dict]:
    """Check if expenses violate defined rules.
    
    Default rule: lunch must be <= $25 per transaction
    """
    violations = []
    
    with _connect() as conn:
        # Get the rule
        cursor = conn.execute(
            """
            SELECT rule_id, max_amount FROM expense_rules
            WHERE rule_name = ?
            """,
            (rule_name,),
        )
        rule = cursor.fetchone()
        
        if not rule:
            return violations
        
        rule_id, max_amount = rule
        
        # Find violated documents if rule exists
        # For lunch limit, check meals category
        cursor = conn.execute(
            """
            SELECT doc_id, vendor, total, doc_date, category
            FROM documents
            WHERE category = 'meals' AND total > ?
            ORDER BY total DESC
            """,
            (max_amount,),
        )
        
        violations = [
            {
                "doc_id": row[0],
                "vendor": row[1],
                "total": row[2],
                "doc_date": row[3],
                "category": row[4],
                "rule_violated": rule_name,
                "overage": row[2] - max_amount,
            }
            for row in cursor.fetchall()
        ]
    
    return violations


def compare_spending_periods(start_date1: str, end_date1: str, start_date2: str, end_date2: str) -> dict:
    """Compare spending across two date periods."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT category, SUM(total) as total
            FROM documents
            WHERE doc_date BETWEEN ? AND ?
            GROUP BY category
            """,
            (start_date1, end_date1),
        )
        period1 = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor = conn.execute(
            """
            SELECT category, SUM(total) as total
            FROM documents
            WHERE doc_date BETWEEN ? AND ?
            GROUP BY category
            """,
            (start_date2, end_date2),
        )
        period2 = {row[0]: row[1] for row in cursor.fetchall()}
    
    comparison = {}
    all_categories = set(list(period1.keys()) + list(period2.keys()))
    
    for category in all_categories:
        p1_total = period1.get(category, 0.0)
        p2_total = period2.get(category, 0.0)
        change = p2_total - p1_total
        change_pct = (change / p1_total * 100) if p1_total > 0 else 0
        
        comparison[category] = {
            "period1": p1_total,
            "period2": p2_total,
            "change": change,
            "change_pct": change_pct,
        }
    
    return comparison


def export_to_csv_format(days: int = None) -> str:
    """Export documents as CSV format string."""
    with _connect() as conn:
        if days:
            cursor = conn.execute(
                """
                SELECT doc_id, vendor, doc_date, category, total, reimbursable, created_at
                FROM documents
                WHERE datetime(created_at) >= datetime('now', '-' || ? || ' days')
                ORDER BY created_at DESC
                """,
                (days,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT doc_id, vendor, doc_date, category, total, reimbursable, created_at
                FROM documents
                ORDER BY created_at DESC
                """
            )
        
        rows = cursor.fetchall()
    
    if not rows:
        return "doc_id,vendor,date,category,total,reimbursable,created_at\n"
    
    csv_lines = ["doc_id,vendor,date,category,total,reimbursable,created_at"]
    for row in rows:
        doc_id, vendor, doc_date, category, total, reimbursable, created_at = row
        total_str = f"{total:.2f}" if isinstance(total, (int, float)) else str(total)
        reimbursable_str = "Yes" if reimbursable else "No"
        csv_lines.append(
            f'{doc_id},"{vendor or ""}",{doc_date or ""},{category or "other"},{total_str},{reimbursable_str},{created_at or ""}'
        )
    
    return "\n".join(csv_lines)


def find_receipts_with_keywords(keywords: list[str]) -> list[dict]:
    """Find receipts containing any of the keywords in description or line items."""
    keyword_pattern = "|".join(keywords)
    
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT doc_id, vendor, doc_date, category, total, line_items, description
            FROM documents
            WHERE line_items LIKE ? OR description LIKE ? OR raw_text LIKE ?
            ORDER BY created_at DESC
            """,
            (f"%{keyword_pattern}%", f"%{keyword_pattern}%", f"%{keyword_pattern}%"),
        )
        return [
            {
                "doc_id": row[0],
                "vendor": row[1],
                "doc_date": row[2],
                "category": row[3],
                "total": row[4],
                "line_items": row[5],
                "description": row[6],
            }
            for row in cursor.fetchall()
        ]


def create_reimbursement_batch(batch_name: str, start_date: str, end_date: str) -> int:
    """Create a reimbursement batch for a date range."""
    with _connect() as conn:
        # Calculate total for the period
        cursor = conn.execute(
            """
            SELECT COALESCE(SUM(total), 0) FROM documents
            WHERE doc_date BETWEEN ? AND ?
            """,
            (start_date, end_date),
        )
        total_amount = cursor.fetchone()[0]
        
        # Insert batch
        cursor = conn.execute(
            """
            INSERT INTO reimbursement_batches (batch_name, start_date, end_date, total_amount)
            VALUES (?, ?, ?, ?)
            """,
            (batch_name, start_date, end_date, total_amount),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_reimbursement_summary(batch_id: int) -> dict:
    """Get summary of a reimbursement batch."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT batch_id, batch_name, start_date, end_date, total_amount, status
            FROM reimbursement_batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        )
        batch = cursor.fetchone()
        if not batch:
            return {}
        
        # Get documents in batch
        cursor = conn.execute(
            """
            SELECT d.doc_id, d.vendor, d.category, d.total, d.doc_date
            FROM documents d
            WHERE d.doc_date BETWEEN ? AND ?
            ORDER BY d.category, d.total DESC
            """,
            (batch[2], batch[3]),
        )
        
        docs = cursor.fetchall()
        category_totals = {}
        for doc in docs:
            category = doc[2] or "other"
            category_totals[category] = category_totals.get(category, 0) + (doc[3] or 0)
        
        return {
            "batch_id": batch[0],
            "batch_name": batch[1],
            "start_date": batch[2],
            "end_date": batch[3],
            "total_amount": batch[4],
            "status": batch[5],
            "category_breakdown": category_totals,
            "document_count": len(docs),
        }


def detect_anomalies() -> list[dict]:
    """Detect suspicious/anomalous invoices."""
    anomalies = []
    
    with _connect() as conn:
        # Find invoices with missing critical fields
        cursor = conn.execute(
            """
            SELECT doc_id, vendor, invoice_number, doc_date, total
            FROM documents
            WHERE invoice_number IS NULL OR doc_date IS NULL
            """
        )
        
        for row in cursor.fetchall():
            doc_id, vendor, invoice_number, doc_date, total = row
            missing = []
            if not invoice_number:
                missing.append("invoice #")
            if not doc_date:
                missing.append("date")
            
            anomalies.append({
                "doc_id": doc_id,
                "vendor": vendor,
                "total": total,
                "anomaly_type": "missing_critical_fields",
                "details": f"Missing: {', '.join(missing)}",
            })
        
        # Find unusually high amounts
        cursor = conn.execute(
            """
            SELECT AVG(total) as avg_total FROM documents
            """
        )
        avg_total = cursor.fetchone()[0] or 0
        threshold = avg_total * 3  # 3x average is anomalous
        
        if threshold > 0:
            cursor = conn.execute(
                """
                SELECT doc_id, vendor, doc_date, total
                FROM documents
                WHERE total > ?
                ORDER BY total DESC
                """,
                (threshold,),
            )
            
            for row in cursor.fetchall():
                doc_id, vendor, doc_date, total = row
                anomalies.append({
                    "doc_id": doc_id,
                    "vendor": vendor,
                    "total": total,
                    "anomaly_type": "unusual_amount",
                    "details": f"Amount ${total:.2f} is {(total/avg_total):.1f}x average",
                })
        
        # Find potential duplicate vendors with different names (fuzzy match)
        cursor = conn.execute(
            """
            SELECT DISTINCT vendor FROM documents
            WHERE vendor IS NOT NULL
            ORDER BY vendor
            """
        )
        
        vendors = [row[0] for row in cursor.fetchall()]
        for i, v1 in enumerate(vendors):
            for v2 in vendors[i+1:]:
                # Simple check: if one contains the other or very similar
                v1_lower = v1.lower().strip()
                v2_lower = v2.lower().strip()
                
                if v1_lower in v2_lower or v2_lower in v1_lower:
                    if v1_lower != v2_lower:
                        anomalies.append({
                            "doc_id": None,
                            "vendor": v1,
                            "total": None,
                            "anomaly_type": "vendor_name_variation",
                            "details": f"Similar vendor names: '{v1}' and '{v2}'",
                        })
                        break
    
    return anomalies


def verify_vendor(vendor_name: str) -> dict:
    """Verify vendor information (stub - would call real API in production)."""
    # This is a stub - in production, would call business registry APIs
    # or vendor verification services
    
    verified_vendors = {
        "McDonald's": {"verified": True, "website": "mcdonalds.com", "type": "Restaurant"},
        "Starbucks": {"verified": True, "website": "starbucks.com", "type": "Coffee Shop"},
        "Uber": {"verified": True, "website": "uber.com", "type": "Transportation"},
        "United Airlines": {"verified": True, "website": "united.com", "type": "Airline"},
        "Home Depot": {"verified": True, "website": "homedepot.com", "type": "Supplies"},
        "Staples": {"verified": True, "website": "staples.com", "type": "Office Supplies"},
    }
    
    # Simple lookup
    for key, info in verified_vendors.items():
        if key.lower() in vendor_name.lower() or vendor_name.lower() in key.lower():
            return {
                "vendor": vendor_name,
                "verified": info["verified"],
                "website": info.get("website", ""),
                "type": info.get("type", ""),
                "confidence": 0.95,
            }
    
    return {
        "vendor": vendor_name,
        "verified": False,
        "website": "",
        "type": "Unknown",
        "confidence": 0.0,
        "note": "Vendor not found in verification database",
    }


def _validate_period(period: str) -> str:
    """Validate and normalize period string."""
    p = period.lower().strip()
    if p not in ("week", "month"):
        raise ValueError(f"Period must be 'week' or 'month', got: {period}")
    return p


def _get_period_label(effective_date: str, period: str) -> str:
    """Generate period label from date string.
    
    Args:
        effective_date: Date string (YYYY-MM-DD format)
        period: 'week' or 'month'
    
    Returns:
        'YYYY-WW' for week, 'YYYY-MM' for month
    """
    if period == "week":
        # strftime('%Y-%W') gives year-weeknumber
        return f"{effective_date[:4]}-W{effective_date[5:7]}"
    else:  # month
        return effective_date[:7]  # YYYY-MM


def spend_by_category_calendar(period: str, n_periods: int = 8) -> list[tuple]:
    """Spending by category for last N calendar periods (weeks or months).
    
    Args:
        period: 'week' or 'month'
        n_periods: Number of periods to return (default 8)
    
    Returns:
        list of (period_label, category, total_spend, count)
        Example: ('2026-W18', 'meals', 145.50, 12)
    """
    period = _validate_period(period)
    
    with _connect() as conn:
        if period == "week":
            cursor = conn.execute(
                """
                SELECT 
                    strftime('%Y-W%W', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label,
                    category,
                    COALESCE(SUM(total), 0) AS total_spend,
                    COUNT(*) AS count
                FROM documents
                GROUP BY period_label, category
                ORDER BY period_label DESC, total_spend DESC
                LIMIT ?
                """,
                (n_periods * 4,),  # More rows since grouped by category too
            )
        else:  # month
            cursor = conn.execute(
                """
                SELECT 
                    strftime('%Y-%m', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label,
                    category,
                    COALESCE(SUM(total), 0) AS total_spend,
                    COUNT(*) AS count
                FROM documents
                GROUP BY period_label, category
                ORDER BY period_label DESC, total_spend DESC
                LIMIT ?
                """,
                (n_periods * 4,),
            )
        
        return cursor.fetchall()


def top_vendors_calendar(period: str, n_periods: int = 8, top_k: int = 5) -> list[tuple]:
    """Top K vendors by spend for last N calendar periods.
    
    Args:
        period: 'week' or 'month'
        n_periods: Number of periods to return
        top_k: Number of top vendors per period (default 5)
    
    Returns:
        list of (period_label, vendor, total_spend)
        Example: ('2026-W18', 'Starbucks', 45.50)
    """
    period = _validate_period(period)
    
    with _connect() as conn:
        if period == "week":
            # Get all vendor spending by week, then filter top K per week
            cursor = conn.execute(
                """
                SELECT 
                    strftime('%Y-W%W', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label,
                    vendor,
                    COALESCE(SUM(total), 0) AS total_spend
                FROM documents
                WHERE vendor IS NOT NULL
                GROUP BY period_label, vendor
                ORDER BY period_label DESC, total_spend DESC
                """
            )
            
            all_rows = cursor.fetchall()
            result = []
            period_vendor_count = {}
            
            for row in all_rows:
                period_label, vendor, total_spend = row
                key = (period_label,)
                
                if key not in period_vendor_count:
                    period_vendor_count[key] = 0
                
                if period_vendor_count[key] < top_k:
                    result.append(row)
                    period_vendor_count[key] += 1
            
            return result[:n_periods * top_k]
        
        else:  # month
            # Get all vendor spending by month, then filter top K per month
            cursor = conn.execute(
                """
                SELECT 
                    strftime('%Y-%m', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label,
                    vendor,
                    COALESCE(SUM(total), 0) AS total_spend
                FROM documents
                WHERE vendor IS NOT NULL
                GROUP BY period_label, vendor
                ORDER BY period_label DESC, total_spend DESC
                """
            )
            
            all_rows = cursor.fetchall()
            result = []
            period_vendor_count = {}
            
            for row in all_rows:
                period_label, vendor, total_spend = row
                key = (period_label,)
                
                if key not in period_vendor_count:
                    period_vendor_count[key] = 0
                
                if period_vendor_count[key] < top_k:
                    result.append(row)
                    period_vendor_count[key] += 1
            
            return result[:n_periods * top_k]


def compare_last_two_periods(period: str) -> dict:
    """Compare spending by category for last period vs previous period.
    
    Args:
        period: 'week' or 'month'
    
    Returns:
        dict with structure:
        {
            "current_period": "2026-W18",
            "previous_period": "2026-W17",
            "comparison": {
                "meals": {"current": 145.50, "previous": 120.00, "change": 25.50, "change_pct": 21.25},
                "travel": {"current": 85.00, "previous": 110.00, "change": -25.00, "change_pct": -22.73}
            }
        }
    """
    period = _validate_period(period)
    
    with _connect() as conn:
        # Get all period labels and sort descending
        if period == "week":
            cursor = conn.execute(
                """
                SELECT DISTINCT strftime('%Y-W%W', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label
                FROM documents
                ORDER BY period_label DESC
                LIMIT 2
                """
            )
        else:  # month
            cursor = conn.execute(
                """
                SELECT DISTINCT strftime('%Y-%m', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label
                FROM documents
                ORDER BY period_label DESC
                LIMIT 2
                """
            )
        
        period_labels = [row[0] for row in cursor.fetchall()]
        
        if len(period_labels) < 2:
            return {
                "error": "Not enough data for comparison (need at least 2 periods)",
                "current_period": period_labels[0] if period_labels else None,
                "previous_period": None,
                "comparison": {},
            }
        
        current_period = period_labels[0]
        previous_period = period_labels[1]
        
        # Get spending by category for current period
        if period == "week":
            cursor = conn.execute(
                """
                SELECT category, COALESCE(SUM(total), 0) AS total
                FROM documents
                WHERE strftime('%Y-W%W', COALESCE(doc_date, substr(created_at, 1, 10))) = ?
                GROUP BY category
                """,
                (current_period,),
            )
        else:  # month
            cursor = conn.execute(
                """
                SELECT category, COALESCE(SUM(total), 0) AS total
                FROM documents
                WHERE strftime('%Y-%m', COALESCE(doc_date, substr(created_at, 1, 10))) = ?
                GROUP BY category
                """,
                (current_period,),
            )
        
        current_data = {row[0] or "other": row[1] for row in cursor.fetchall()}
        
        # Get spending by category for previous period
        if period == "week":
            cursor = conn.execute(
                """
                SELECT category, COALESCE(SUM(total), 0) AS total
                FROM documents
                WHERE strftime('%Y-W%W', COALESCE(doc_date, substr(created_at, 1, 10))) = ?
                GROUP BY category
                """,
                (previous_period,),
            )
        else:  # month
            cursor = conn.execute(
                """
                SELECT category, COALESCE(SUM(total), 0) AS total
                FROM documents
                WHERE strftime('%Y-%m', COALESCE(doc_date, substr(created_at, 1, 10))) = ?
                GROUP BY category
                """,
                (previous_period,),
            )
        
        previous_data = {row[0] or "other": row[1] for row in cursor.fetchall()}
    
    # Build comparison
    all_categories = set(list(current_data.keys()) + list(previous_data.keys()))
    comparison = {}
    
    for category in sorted(all_categories):
        current = current_data.get(category, 0.0)
        previous = previous_data.get(category, 0.0)
        change = current - previous
        change_pct = (change / previous * 100) if previous > 0 else (100.0 if current > 0 else 0.0)
        
        comparison[category] = {
            "current": round(current, 2),
            "previous": round(previous, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
        }
    
    return {
        "current_period": current_period,
        "previous_period": previous_period,
        "comparison": comparison,
    }


def list_pending_receipts(limit: int = 50) -> list[dict]:
    """List receipts with incomplete or pending data.
    
    A receipt is pending if vendor/date/total is NULL/empty OR is_pending=1.
    
    Returns:
        list of dicts with: doc_id, vendor, doc_date, total, currency, category,
        missing_fields (comma-separated), created_at
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT 
                doc_id, vendor, doc_date, total, currency, category, created_at, is_pending,
                CASE 
                    WHEN vendor IS NULL OR vendor = '' THEN 1 ELSE 0 
                END as missing_vendor,
                CASE 
                    WHEN doc_date IS NULL OR doc_date = '' THEN 1 ELSE 0 
                END as missing_date,
                CASE 
                    WHEN total IS NULL OR total = 0 THEN 1 ELSE 0 
                END as missing_total
            FROM documents
            WHERE is_pending = 1 
               OR vendor IS NULL OR vendor = ''
               OR doc_date IS NULL OR doc_date = ''
               OR total IS NULL OR total = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        
        results = []
        for row in cursor.fetchall():
            doc_id, vendor, doc_date, total, currency, category, created_at, is_pending, \
                missing_vendor, missing_date, missing_total = row
            
            # Build missing_fields list
            missing = []
            if missing_vendor:
                missing.append("vendor")
            if missing_date:
                missing.append("date")
            if missing_total:
                missing.append("total")
            
            results.append({
                "doc_id": doc_id,
                "vendor": vendor,
                "doc_date": doc_date,
                "total": total,
                "currency": currency,
                "category": category,
                "missing_fields": ", ".join(missing) if missing else "none",
                "created_at": created_at,
            })
    
    return results


def get_document(doc_id: int) -> dict:
    """Retrieve a full document by ID.
    
    Returns: dict with all document fields
    """
    return get_document_by_id(doc_id)


def update_document(doc_id: int, updates: dict) -> None:
    """Update document fields.
    
    Allowed fields: vendor, doc_date, total, tax, subtotal, currency, category,
                    invoice_number, description, is_pending
    Sets updated_at to current timestamp.
    
    If vendor and category are both provided, learns the association in vendor_profiles.
    """
    allowed_fields = {
        "vendor", "doc_date", "total", "tax", "subtotal", "currency",
        "category", "invoice_number", "description", "is_pending"
    }
    
    # Filter updates to only allowed fields
    filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
    
    if not filtered_updates:
        return
    
    # Learn vendor→category association if both are present
    vendor = filtered_updates.get("vendor")
    category = filtered_updates.get("category")
    if vendor and category:
        set_vendor_category(vendor, category)
    
    # Build SET clause dynamically
    set_clauses = [f"{field} = ?" for field in filtered_updates.keys()]
    set_clauses.append("updated_at = datetime('now')")
    set_clause = ", ".join(set_clauses)
    
    # Build values tuple (add doc_id at the end for WHERE clause)
    values = tuple(filtered_updates.values()) + (doc_id,)
    
    with _connect() as conn:
        conn.execute(
            f"""
            UPDATE documents
            SET {set_clause}
            WHERE doc_id = ?
            """,
            values,
        )
        conn.commit()


def get_vendor_category(vendor: str) -> str | None:
    """Retrieve the learned category for a vendor from vendor_profiles.
    
    Args:
        vendor: Vendor name to look up
    
    Returns:
        Category name if vendor found in vendor_profiles, None otherwise
    """
    if not vendor:
        return None
    
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT category FROM vendor_profiles
            WHERE LOWER(vendor) = LOWER(?)
            """,
            (vendor,),
        )
        row = cursor.fetchone()
        return row[0] if row else None


def set_vendor_category(vendor: str, category: str) -> None:
    """Associate a vendor with a category in vendor_profiles.
    
    Creates or updates vendor profile entry.
    """
    with _connect() as conn:
        # Check if vendor already exists
        cursor = conn.execute(
            """
            SELECT vendor FROM vendor_profiles WHERE vendor = ?
            """,
            (vendor,),
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            conn.execute(
                """
                UPDATE vendor_profiles
                SET category = ?, updated_at = datetime('now')
                WHERE vendor = ?
                """,
                (category, vendor),
            )
        else:
            # Insert new
            conn.execute(
                """
                INSERT INTO vendor_profiles (vendor, category, updated_at)
                VALUES (?, ?, datetime('now'))
                """,
                (vendor, category),
            )
        
        conn.commit()


def get_categories() -> list[str]:
    """Get list of active categories.
    
    Returns:
        list of category names (strings)
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT category FROM categories
            WHERE is_active = 1
            ORDER BY category
            """
        )
        return [row[0] for row in cursor.fetchall()]


def add_category(category: str, display_name: str | None = None) -> None:
    """Add a new category.
    
    Args:
        category: Category name (internal identifier)
        display_name: Human-readable display name (defaults to category name)
    """
    display_name = display_name or category
    
    with _connect() as conn:
        # Check if already exists
        cursor = conn.execute(
            """
            SELECT category FROM categories WHERE category = ?
            """,
            (category,),
        )
        
        if cursor.fetchone():
            # Already exists, update display_name and activate
            conn.execute(
                """
                UPDATE categories
                SET display_name = ?, is_active = 1
                WHERE category = ?
                """,
                (display_name, category),
            )
        else:
            # Insert new
            conn.execute(
                """
                INSERT INTO categories (category, display_name, is_active)
                VALUES (?, ?, 1)
                """,
                (category, display_name),
            )
        
        conn.commit()
