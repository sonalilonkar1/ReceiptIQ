"""Initialize local database schema for ReceiptIQ with categories and vendor profiles."""

from pathlib import Path
import sqlite3


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table (idempotent migration check)."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add column to table only if it doesn't exist (idempotent)."""
    if not column_exists(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        print(f"  ✓ Added column {table}.{column}")


def create_base_tables(conn: sqlite3.Connection) -> None:
    """Create base tables if not exist."""
    
    # Documents table
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

    # Audit flags table
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

    # Expense rules table
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

    # Reimbursement batches table
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

    # Batch documents mapping table
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


def create_new_tables(conn: sqlite3.Connection) -> None:
    """Create new tables for categories and vendor profiles."""
    
    # Categories table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            category TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Vendor profiles table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vendor_profiles (
            vendor TEXT PRIMARY KEY,
            category TEXT,
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (category) REFERENCES categories(category)
        );
        """
    )


def migrate_documents_table(conn: sqlite3.Connection) -> None:
    """Add missing columns to documents table (idempotent)."""
    add_column_if_missing(conn, "documents", "is_pending", "INTEGER DEFAULT 0")
    add_column_if_missing(conn, "documents", "updated_at", "TEXT DEFAULT (datetime('now'))")


def seed_default_categories(conn: sqlite3.Connection) -> None:
    """Seed default categories if table is empty."""
    cursor = conn.execute("SELECT COUNT(*) FROM categories")
    count = cursor.fetchone()[0]
    
    if count == 0:
        default_categories = [
            ("meals", "Meals & Food", 1),
            ("travel", "Travel & Transport", 1),
            ("supplies", "Office Supplies", 1),
            ("other", "Other", 1),
        ]
        conn.executemany(
            "INSERT INTO categories (category, display_name, is_active) VALUES (?, ?, ?)",
            default_categories
        )
        print(f"  ✓ Seeded {len(default_categories)} default categories")


def init_db() -> None:
    """Initialize database with all tables and migrations."""
    db_path = Path(__file__).resolve().parents[1] / "receiptiq.sqlite"

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Create base tables
        create_base_tables(conn)
        
        # Create new tables (categories, vendor_profiles)
        create_new_tables(conn)
        
        # Migrate existing documents table (add missing columns)
        migrate_documents_table(conn)
        
        # Seed default categories
        seed_default_categories(conn)

        conn.commit()

    print(f"✓ Database initialized at: {db_path}")


if __name__ == "__main__":
    init_db()
