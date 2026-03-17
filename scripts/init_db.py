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

        conn.commit()

    print(f"Database initialized at: {db_path}")


if __name__ == "__main__":
    init_db()
