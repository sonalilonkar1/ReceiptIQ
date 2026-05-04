#!/usr/bin/env python3
"""Test script to verify calendar grouping in SQLite.

Inserts test documents with known dates and verifies that:
- Week grouping produces 'YYYY-WW' format
- Month grouping produces 'YYYY-MM' format
- NULL dates fall back to created_at
"""

from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

# Get DB path from project root
DB_PATH = Path(__file__).resolve().parents[1] / "receiptiq.sqlite"


def _connect():
    """Connect to database."""
    return sqlite3.connect(DB_PATH)


def test_calendar_grouping():
    """Insert test documents and verify calendar grouping."""
    print("=" * 70)
    print("CALENDAR GROUPING TEST")
    print("=" * 70)
    
    # Test dates covering multiple weeks and months
    today = datetime(2026, 5, 3)  # May 3, 2026 (week 18, month 05)
    test_dates = [
        (today, "meal_today"),  # Week 18, May
        (today - timedelta(days=7), "meal_last_week"),  # Week 17, April
        (today - timedelta(days=35), "meal_5_weeks_ago"),  # Week 13, March
    ]
    
    with _connect() as conn:
        # Insert test documents
        print("\n📝 Inserting test documents...")
        doc_ids = []
        for test_date, vendor_name in test_dates:
            doc_date_str = test_date.strftime("%Y-%m-%d")
            
            cursor = conn.execute(
                """
                INSERT INTO documents (
                    doc_type, vendor, doc_date, currency, total, 
                    confidence, category, description, raw_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "receipt",
                    vendor_name,
                    doc_date_str,
                    "USD",
                    25.00 + len(doc_ids) * 5,  # Incrementing totals
                    0.95,
                    "meals",
                    f"Test receipt for {vendor_name}",
                    f"Test raw text for {vendor_name}",
                ),
            )
            doc_ids.append(cursor.lastrowid)
            print(f"  ✓ Inserted doc_id={cursor.lastrowid}, vendor={vendor_name}, date={doc_date_str}")
        
        conn.commit()
        
        # Test week grouping
        print("\n📊 Testing WEEK grouping...")
        print("  Query: SELECT strftime('%Y-W%W', COALESCE(doc_date, substr(created_at,1,10))) AS period_label, ...")
        cursor = conn.execute(
            """
            SELECT 
                strftime('%Y-W%W', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label,
                vendor,
                doc_date,
                COALESCE(SUM(total), 0) AS total_spend,
                COUNT(*) AS count
            FROM documents
            WHERE vendor LIKE 'meal_%'
            GROUP BY period_label, vendor
            ORDER BY period_label DESC
            """
        )
        
        week_results = cursor.fetchall()
        print(f"\n  Results ({len(week_results)} rows):")
        print(f"  {'Period Label':<15} {'Vendor':<20} {'Doc Date':<12} {'Total':<10} {'Count':<6}")
        print("  " + "-" * 65)
        
        for row in week_results:
            period_label, vendor, doc_date, total, count = row
            print(f"  {period_label:<15} {vendor:<20} {doc_date or '(NULL)':<12} ${total:>8.2f} {count:>5}")
        
        # Verify format
        print("\n  ✓ Format verification:")
        for row in week_results:
            period_label = row[0]
            if period_label:
                parts = period_label.split('-W')
                if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 2:
                    print(f"    ✓ '{period_label}' matches 'YYYY-WW' format")
                else:
                    print(f"    ✗ '{period_label}' does NOT match 'YYYY-WW' format")
        
        # Test month grouping
        print("\n📊 Testing MONTH grouping...")
        print("  Query: SELECT strftime('%Y-%m', COALESCE(doc_date, substr(created_at,1,10))) AS period_label, ...")
        cursor = conn.execute(
            """
            SELECT 
                strftime('%Y-%m', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label,
                vendor,
                doc_date,
                COALESCE(SUM(total), 0) AS total_spend,
                COUNT(*) AS count
            FROM documents
            WHERE vendor LIKE 'meal_%'
            GROUP BY period_label, vendor
            ORDER BY period_label DESC
            """
        )
        
        month_results = cursor.fetchall()
        print(f"\n  Results ({len(month_results)} rows):")
        print(f"  {'Period Label':<15} {'Vendor':<20} {'Doc Date':<12} {'Total':<10} {'Count':<6}")
        print("  " + "-" * 65)
        
        for row in month_results:
            period_label, vendor, doc_date, total, count = row
            print(f"  {period_label:<15} {vendor:<20} {doc_date or '(NULL)':<12} ${total:>8.2f} {count:>5}")
        
        # Verify format
        print("\n  ✓ Format verification:")
        for row in month_results:
            period_label = row[0]
            if period_label:
                parts = period_label.split('-')
                if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 2:
                    print(f"    ✓ '{period_label}' matches 'YYYY-MM' format")
                else:
                    print(f"    ✗ '{period_label}' does NOT match 'YYYY-MM' format")
        
        # Test NULL date handling
        print("\n📊 Testing NULL date handling...")
        print("  Inserting document with NULL doc_date...")
        
        cursor = conn.execute(
            """
            INSERT INTO documents (
                doc_type, vendor, doc_date, currency, total, 
                confidence, category, description, raw_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "receipt",
                "null_date_vendor",
                None,  # NULL date
                "USD",
                50.00,
                0.95,
                "meals",
                "Test receipt with NULL date",
                "Test raw text",
            ),
        )
        conn.commit()
        
        # Query with NULL date
        cursor = conn.execute(
            """
            SELECT 
                strftime('%Y-%m', COALESCE(doc_date, substr(created_at, 1, 10))) AS period_label,
                vendor,
                doc_date,
                created_at,
                total
            FROM documents
            WHERE vendor = 'null_date_vendor'
            """
        )
        
        null_result = cursor.fetchone()
        if null_result:
            period_label, vendor, doc_date, created_at, total = null_result
            print(f"  ✓ Vendor: {vendor}")
            print(f"    doc_date: {doc_date}")
            print(f"    created_at: {created_at}")
            print(f"    period_label (using created_at): {period_label}")
            
            # Parse date from created_at
            created_date = created_at.split(' ')[0]  # Get YYYY-MM-DD part
            expected_period = created_date[:7]  # Get YYYY-MM
            if period_label == expected_period:
                print(f"    ✓ NULL date correctly falls back to created_at")
            else:
                print(f"    ✗ Period label mismatch: expected {expected_period}, got {period_label}")
        
        # Clean up test data
        print("\n🧹 Cleaning up test data...")
        conn.execute("DELETE FROM documents WHERE vendor LIKE 'meal_%' OR vendor = 'null_date_vendor'")
        conn.commit()
        print("  ✓ Test documents removed")
    
    print("\n" + "=" * 70)
    print("✅ CALENDAR GROUPING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_calendar_grouping()
