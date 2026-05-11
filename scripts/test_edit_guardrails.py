#!/usr/bin/env python3
"""
Validates professor's concern: user edits are untrusted and protected by guardrails.
Tests: invalid field validation, totals mismatch detection, audit trails, and pending status clearing.
"""

import sys
from pathlib import Path

# Add app to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.tools.db import (
    _connect,
    insert_document,
    get_document_by_id,
    update_document,
    get_audit_flags,
    set_vendor_category,
    add_flag,
)


def init_database():
    """Ensure database is initialized with proper schema."""
    # Import after path is set
    from scripts.init_db import init_db
    init_db()
    print("✓ Database initialized")


def ensure_dummy_document():
    """Insert a dummy document if DB is empty."""
    with _connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
    
    if count == 0:
        dummy = {
            "doc_type": "invoice",
            "vendor": "Test Vendor Inc",
            "doc_date": "2026-05-10",
            "currency": "USD",
            "subtotal": 50.0,
            "tax": 5.0,
            "total": 55.0,
            "confidence": 0.95,
            "category": "meals",
            "line_items": "test item 1\ntest item 2",
            "description": "Dummy receipt for testing",
            "raw_text": "Test receipt",
        }
        doc_id = insert_document(dummy)
        print(f"✓ Inserted dummy document (doc_id={doc_id})")
        return doc_id
    else:
        print(f"✓ Database has {count} document(s)")
        # Get first doc_id
        with _connect() as conn:
            cursor = conn.execute("SELECT doc_id FROM documents LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else None


def test_invalid_total_string():
    """Test: total = "abc" should be rejected."""
    print("\n[TEST 1] Invalid total (string)")
    doc_id = ensure_dummy_document()
    
    try:
        update_document(doc_id, {"total": "abc"})
        print("  ❌ FAIL: String total was accepted (should reject)")
        return False
    except ValueError as e:
        if "numeric" in str(e).lower():
            print(f"  ✅ PASS: Rejected with error: {e}")
            return True
        else:
            print(f"  ❌ FAIL: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected error: {e}")
        return False


def test_invalid_total_negative():
    """Test: total = -5 should be rejected."""
    print("\n[TEST 2] Invalid total (negative)")
    doc_id = ensure_dummy_document()
    
    try:
        update_document(doc_id, {"total": -5})
        print("  ❌ FAIL: Negative total was accepted (should reject)")
        return False
    except ValueError as e:
        if "negative" in str(e).lower():
            print(f"  ✅ PASS: Rejected with error: {e}")
            return True
        else:
            print(f"  ❌ FAIL: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected error: {e}")
        return False


def test_invalid_currency():
    """Test: currency = "XYZ" should be rejected."""
    print("\n[TEST 3] Invalid currency")
    doc_id = ensure_dummy_document()
    
    try:
        update_document(doc_id, {"currency": "XYZ"})
        print("  ❌ FAIL: Invalid currency was accepted (should reject)")
        return False
    except ValueError as e:
        if "unsupported" in str(e).lower() or "currency" in str(e).lower():
            print(f"  ✅ PASS: Rejected with error: {e}")
            return True
        else:
            print(f"  ❌ FAIL: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected error: {e}")
        return False


def test_invalid_date():
    """Test: doc_date = "13/40/2026" should be rejected."""
    print("\n[TEST 4] Invalid date")
    doc_id = ensure_dummy_document()
    
    try:
        update_document(doc_id, {"doc_date": "13/40/2026"})
        print("  ❌ FAIL: Invalid date was accepted (should reject)")
        return False
    except ValueError as e:
        if "date" in str(e).lower():
            print(f"  ✅ PASS: Rejected with error: {e}")
            return True
        else:
            print(f"  ❌ FAIL: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected error: {e}")
        return False


def test_totals_mismatch():
    """Test: totals mismatch (subtotal=10, tax=2, total=50) creates audit flag."""
    print("\n[TEST 5] Totals mismatch detection")
    doc_id = ensure_dummy_document()
    
    try:
        # This should succeed (soft guardrail) but create audit flag
        update_document(doc_id, {"subtotal": 10, "tax": 2, "total": 50})
        
        # Check if audit flag was created
        flags = get_audit_flags(doc_id)
        totals_flag = [f for f in flags if f["flag_type"] == "totals_validation"]
        
        if totals_flag:
            print(f"  ✅ PASS: Mismatch detected, audit flag created: {totals_flag[0]['detail']}")
            return True
        else:
            print(f"  ❌ FAIL: Update succeeded but no totals_validation flag created")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected error: {e}")
        return False


def test_valid_edit_and_pending_clear():
    """Test: valid edit clears is_pending if vendor/date/total present."""
    print("\n[TEST 6] Valid edit and is_pending auto-clear")
    
    # Create a new pending document
    pending_doc = {
        "doc_type": "invoice",
        "vendor": None,
        "doc_date": None,
        "currency": "USD",
        "subtotal": None,
        "tax": None,
        "total": None,
        "confidence": 0.5,
        "category": None,
        "description": "Pending receipt",
        "raw_text": "incomplete",
        "is_pending": 1,
    }
    doc_id = insert_document(pending_doc)
    
    # Verify is_pending is set
    doc = get_document_by_id(doc_id)
    if doc.get("is_pending") != 1:
        print(f"  ⚠ Warning: is_pending not set (got {doc.get('is_pending')})")
    
    try:
        # Update with vendor, date, total
        update_document(doc_id, {
            "vendor": "Coffee Shop",
            "doc_date": "2026-05-10",
            "total": 12.50,
        })
        
        # Check if is_pending was cleared
        updated_doc = get_document_by_id(doc_id)
        if updated_doc.get("is_pending") == 0 or updated_doc.get("is_pending") is None:
            print(f"  ✅ PASS: Document updated and is_pending cleared (now: {updated_doc.get('is_pending')})")
            
            # Verify audit trail
            flags = get_audit_flags(doc_id)
            if len(flags) > 0:
                print(f"     Audit flags created: {len(flags)}")
            return True
        else:
            print(f"  ❌ FAIL: is_pending not cleared (still {updated_doc.get('is_pending')})")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected error: {e}")
        return False


def test_vendor_profile_mapping():
    """Test: vendor+category creates vendor_profiles entry."""
    print("\n[TEST 7] Vendor profile mapping")
    
    # Create document with vendor and category
    doc = {
        "doc_type": "invoice",
        "vendor": "Unique Vendor ABC",
        "doc_date": "2026-05-10",
        "currency": "USD",
        "total": 100.0,
        "category": "travel",
        "confidence": 0.9,
    }
    doc_id = insert_document(doc)
    
    try:
        # Update with vendor + category
        update_document(doc_id, {
            "vendor": "Updated Vendor XYZ",
            "category": "supplies",
        })
        
        # Verify vendor_profiles entry was created/updated
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT category FROM vendor_profiles WHERE vendor = ?",
                ("Updated Vendor XYZ",)
            )
            row = cursor.fetchone()
        
        if row and row[0] == "supplies":
            print(f"  ✅ PASS: Vendor profile created/mapped (vendor→{row[0]})")
            return True
        else:
            print(f"  ❌ FAIL: Vendor profile not found or wrong category (got {row})")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected error: {e}")
        return False


def test_audit_trail():
    """Test: document_edits table records all changes."""
    print("\n[TEST 8] Audit trail (document_edits)")
    
    # Create document
    doc = {
        "doc_type": "invoice",
        "vendor": "Original Vendor",
        "doc_date": "2026-01-01",
        "currency": "USD",
        "total": 25.0,
        "category": "meals",
        "confidence": 0.8,
    }
    doc_id = insert_document(doc)
    
    try:
        # Make an edit
        update_document(doc_id, {
            "vendor": "Modified Vendor",
            "total": 30.0,
        })
        
        # Check document_edits table
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT field, old_value, new_value FROM document_edits WHERE doc_id = ? ORDER BY edit_id",
                (doc_id,)
            )
            edits = cursor.fetchall()
        
        if len(edits) >= 2:
            print(f"  ✅ PASS: {len(edits)} edits recorded in audit trail")
            for field, old_val, new_val in edits:
                print(f"     {field}: {old_val} → {new_val}")
            return True
        else:
            print(f"  ❌ FAIL: Expected >= 2 edits, got {len(edits)}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected error: {e}")
        return False


def test_valid_date_formats():
    """Test: various valid date formats are accepted."""
    print("\n[TEST 9] Valid date format normalization")
    
    doc_id = ensure_dummy_document()
    test_dates = [
        ("05/10/2026", "2026-05-10"),
        ("2026-05-10", "2026-05-10"),
        ("05-10-2026", "2026-05-10"),
    ]
    
    all_pass = True
    for input_date, expected in test_dates:
        try:
            update_document(doc_id, {"doc_date": input_date})
            updated = get_document_by_id(doc_id)
            if updated.get("doc_date") == expected:
                print(f"  ✅ '{input_date}' → '{expected}'")
            else:
                print(f"  ❌ '{input_date}' → '{updated.get('doc_date')}' (expected {expected})")
                all_pass = False
        except Exception as e:
            print(f"  ❌ Failed to parse '{input_date}': {e}")
            all_pass = False
    
    return all_pass


def main():
    """Run all tests and print summary."""
    print("=" * 70)
    print("TESTING EDIT GUARDRAILS - Professor's Concern Validation")
    print("=" * 70)
    
    # Setup
    init_database()
    ensure_dummy_document()
    
    # Run tests
    results = []
    results.append(("Invalid total (string)", test_invalid_total_string()))
    results.append(("Invalid total (negative)", test_invalid_total_negative()))
    results.append(("Invalid currency", test_invalid_currency()))
    results.append(("Invalid date", test_invalid_date()))
    results.append(("Totals mismatch detection", test_totals_mismatch()))
    results.append(("Valid edit + pending clear", test_valid_edit_and_pending_clear()))
    results.append(("Vendor profile mapping", test_vendor_profile_mapping()))
    results.append(("Audit trail (document_edits)", test_audit_trail()))
    results.append(("Date format normalization", test_valid_date_formats()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} | {name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All guardrails working correctly!")
    else:
        print(f"⚠ {total - passed} test(s) failed - review guardrails")
    print("=" * 70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
