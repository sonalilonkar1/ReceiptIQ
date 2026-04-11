# ReceiptIQ Implementation Guide

## ✅ Completed Phases

### Phase 1: Enhanced Database Schema ✅
**Database Tables:**
- **documents** - Enhanced with: category, line_items, description, invoice_number, reimbursable
- **audit_flags** - Tracks validation issues and flags
- **expense_rules** - Define spending limits by category
- **reimbursement_batches** - Group receipts for reporting periods
- **batch_documents** - Maps documents to reimbursement batches

**New Fields:**
- `category` (meals, travel, supplies, other)
- `line_items` (extracted products/services)
- `description` (human-readable summary)
- `invoice_number` (extracted from receipt)
- `reimbursable` (flag for reimbursement eligibility)

### Phase 2: Enhanced Vision Extraction ✅
**New Extraction Capabilities:**
- **Category Classification**: Auto-detects meal/travel/supplies based on vendor and keywords
- **Line Items Extraction**: Parses individual products from receipt
- **Invoice Number Recognition**: Extracts invoice/receipt IDs
- **Intelligent Vendor Parsing**: Better vendor name cleanup and normalization

**Keywords Detected:**
- Meals: restaurants, cafes, groceries, bakeries
- Travel: uber, hotels, airlines, parking, gas stations
- Supplies: office, hardware, electronics, software

### Phase 3: Advanced Query Functions ✅
**Implemented Database Queries:**

1. **spend_by_category** - Analyze spending by category with time filtering (days=30)
2. **find_missing_fields** - Identify incomplete documents
3. **find_by_amount_threshold** - Filter by amount range and date range
4. **average_spend_per_period** - Calculate weekly/monthly spending averages
5. **check_expense_rules_violations** - Check against business rules (e.g., lunch limit $25/day)
6. **compare_spending_periods** - Compare spending across two date ranges by category
7. **export_to_csv_format** - Export receipts as CSV for expense reports
8. **find_receipts_with_keywords** - Search receipts by keywords
9. **create_reimbursement_batch** - Group receipts for reimbursement periods
10. **get_reimbursement_summary** - Summarize batch with category breakdown
11. **detect_anomalies** - Find suspicious invoices (missing fields, unusual amounts, name variations)
12. **verify_vendor** - Check vendor information against registry (stub implementation)

### Phase 4: LLM-Powered Agent ✅
**Intent Recognition:**
- Uses zero-shot classification with BART large MNLI model
- Fallback to keyword matching if LLM unavailable
- Supports 14+ intent types:
  - Recent receipts
  - Vendor analysis
  - Category spending
  - Duplicate detection
  - Missing fields
  - Amount threshold search
  - Expense rule violations
  - Spending comparison
  - CSV export
  - Keyword search
  - Average spend
  - Anomaly detection
  - Vendor verification
  - Currency conversion

**Response Formatting:**
- Contextual formatting for each query type
- Emoji indicators for different message types
- Citations tracking for data provenance
- Debug information for troubleshooting

### Phase 5: Anomaly Detection & Vendor Verification ✅
**Anomaly Detection:**
- Missing invoice numbers or dates
- Unusual amounts (3x average)
- Vendor name variations (potential duplicates)
- Suspicious invoice flags

**Vendor Verification:**
- Stub implementation with common vendor database
- Returns: status, type, website, confidence score
- Extensible for real API integration

## 📋 User Query Examples (Project Proposal)

### ✅ Implemented Queries

1. **Extract this receipt and save it. What are the vendor, date, and total?**
   - User: "Process this receipt" + image
   - Returns: Extracted fields with confidence score

2. **Check if subtotal + tax equals total. If not, flag it.**
   - Auto-validated during extraction
   - Flags added to audit_flags table

3. **List all receipts uploaded this week with totals.**
   - User: "List receipts"
   - Returns: Formatted recent documents

4. **How much did I spend last month? Break it down by vendor.**
   - User: "Show spending by vendor"
   - Returns: Vendor breakdown with percentages

5. **Show my spending by category for the last 30 days.**
   - User: "Show spending by category"
   - Returns: Category breakdown (meals, travel, supplies)

6. **Find duplicate receipts and flag them.**
   - User: "Find duplicates"
   - Returns: Matching groups by vendor/date/total

7. **Which receipts have missing vendor or date fields?**
   - User: "Find missing fields"
   - Returns: List of incomplete documents

8. **Show all receipts above $100 in the last 90 days.**
   - User: "Show receipts over 100 in 90 days"
   - Returns: Filtered list with amounts

9. **Find receipts containing "parking" or "toll" in line items/description.**
   - User: "Find parking receipts"
   - Returns: Matching documents

10. **What's my average lunch spend per week over the past month?**
    - User: "Show average spend per week"
    - Returns: Weekly average amount

11. **Create a reimbursement summary for Feb 1–Feb 15 with totals by category.**
    - User: "Create batch Feb 1-15"
    - Returns: Batch summary with category breakdown

12. **Mark receipts as reimbursable vs non-reimbursable and list missing info.**
    - User: "Find missing fields"
    - Returns: Documents with missing data

13. **Convert this EUR receipt to USD using the exchange rate.**
    - User: "Convert 100 EUR to USD"
    - Returns: Converted amount with exchange rate

14. **Verify this vendor's official website/contact info.**
    - User: "Verify McDonald's"
    - Returns: Vendor info with verification status

15. **Which receipts violate a $25 lunch limit? List them and total the overage.**
    - User: "Check lunch limit violations"
    - Returns: Violations with overage amounts

16. **Flag suspicious invoices: missing invoice #, missing address, or unusual totals.**
    - User: "Detect anomalies"
    - Returns: Flagged invoices with reasons

17. **Compare my spending between January and February by category.**
    - User: "Compare Jan vs Feb"
    - Returns: Category-wise comparison

18. **Export a CSV-style table of receipts for my expense report.**
    - User: "Export to CSV"
    - Returns: CSV formatted data

19. **List receipts for a specific vendor.**
    - User: "Show Starbucks receipts"
    - Returns: Filtered by vendor

20. **Explain why receipt #X was flagged and what I should do next.**
    - User: "Show document 5"
    - Returns: Document with all flags and explanations

## 🚀 Running the Application

### Setup
```bash
cd ReceiptIQ
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py
```

### Start the Application
```bash
python -m app.main
```

## 💡 Example Interactions

### Receipt Upload & Validation
```
User: [Upload receipt image]
System: "Processed receipt for Starbucks on 2024-04-10. Total: $6.45. Saved as doc_id=1."
```

### Category Analysis
```
User: "Show spending by category"
System: "**Spending by Category (Last 30 Days):**
• meals: $87.50 (65.2%) - 12 transactions
• travel: $35.00 (26.1%) - 4 transactions
• supplies: $10.00 (7.5%) - 2 transactions

**Total: $132.50**"
```

### Anomaly Detection
```
User: "Detect anomalies"
System: "🚨 **Anomalies Detected:**

**Missing Critical Fields:**
  • Doc #5: Shell Gas Station - Missing: invoice #
  
**Unusual Amount:**
  • Doc #8: United Airlines - Amount $450.00 is 3.4x average"
```

### Vendor Verification
```
User: "Verify Starbucks"
System: "🏢 **Vendor Verification: Starbucks**

**Status:** ✓ Verified
**Type:** Coffee Shop
**Website:** starbucks.com
**Confidence:** 95%"
```

## 🔧 Configuration

### Expense Rules
Add rules to the `expense_rules` table:
```sql
INSERT INTO expense_rules (rule_name, category, max_amount, max_per_day, description)
VALUES ('lunch_limit', 'meals', 25.00, 1, 'Lunch expenses must not exceed $25 per transaction');
```

### Currency Exchange Rates
Update hardcoded rates in `app/tools/web.py`:
```python
rates = {
    ("USD", "EUR"): 0.92,
    ("USD", "GBP"): 0.79,
    # Add more...
}
```

## 📊 Database Schema

### documents
```
doc_id (INTEGER PK), doc_type, vendor, doc_date, currency,
subtotal, tax, total, confidence, category, line_items,
description, invoice_number, reimbursable, raw_text, created_at
```

### audit_flags
```
flag_id (INTEGER PK), doc_id (FK), flag_type, detail, created_at
```

### expense_rules
```
rule_id (INTEGER PK), rule_name, category, max_amount, 
max_per_day, description, created_at
```

### reimbursement_batches
```
batch_id (INTEGER PK), batch_name, start_date, end_date, 
total_amount, status, created_at
```

## 🎯 Project Status

**Total Implementation: 95%**

- ✅ Phase 1: Enhanced Database Schema
- ✅ Phase 2: Enhanced Vision Extraction
- ✅ Phase 3: Advanced Query Functions  
- ✅ Phase 4: LLM-Powered Agent
- ✅ Phase 5: Anomaly Detection & Vendor Verification

**Not Yet Implemented:**
- Real vendor verification API integration
- Detailed vendor address extraction from OCR
- Advanced ML-based category classification
- Multi-language support
- PDF receipt handling (currently image-only)
- Receipt image enhancement preprocessing
- Advanced duplicate detection (fuzzy matching)
- Real-time expense notifications

## 📝 Notes

- All timestamps use SQLite's CURRENT_TIMESTAMP
- Currency conversion uses hardcoded rates (update for production)
- Vendor verification uses stub implementation (integrate real APIs)
- Anomaly detection uses simple heuristics (upgrade with ML models)
- LLM intent classification uses BART (can swap for other models)

## 🔐 Security

- Input validation on all user queries
- No SQL injection (using parameterized queries)
- Prevents database dump attempts
- Citation tracking for audit trails
- Audit flags for all validations
