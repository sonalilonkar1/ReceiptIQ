"""Initialize local database schema for ReceiptIQ."""

from pathlib import Path
import sqlite3


def init_db() -> None:
    db_path = Path(__file__).resolve().parents[1] / "receiptiq.sqlite"

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_type TEXT,
                vendor TEXT,
                doc_date TEXT,
                currency TEXT,
                subtotal REAL,
                tax REAL,
                total REAL,
                confidence REAL,
                category TEXT,
                line_items TEXT,
                description TEXT,
                reimbursable INTEGER DEFAULT 0,
                invoice_number TEXT,
                raw_text TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_flags (
                flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                flag_type TEXT,
                detail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expense_rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT,
                category TEXT,
                max_amount REAL,
                max_per_day INTEGER DEFAULT 1,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reimbursement_batches (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_name TEXT,
                start_date TEXT,
                end_date TEXT,
                total_amount REAL,
                status TEXT DEFAULT 'draft',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_documents (
                mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER,
                doc_id INTEGER,
                FOREIGN KEY (batch_id) REFERENCES reimbursement_batches(batch_id),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            );
            """
        )

        conn.commit()

    print(f"Database initialized at: {db_path}")


if __name__ == "__main__":
    init_db()
