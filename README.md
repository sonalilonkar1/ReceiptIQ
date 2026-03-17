# ReceiptIQ

ReceiptIQ is a Python project scaffold for building receipt intelligence workflows with OCR, document parsing, web/data tools, and LLM-powered orchestration.

## Project Structure

```text
ReceiptIQ/
  requirements.txt
  README.md
  app/
    __init__.py
    main.py
    agent.py
    tools/
      __init__.py
      vision.py
      db.py
      web.py
    prompts/
      system.txt
  scripts/
    init_db.py
```

## Quickstart (Local)

Run all commands from the ReceiptIQ project root.

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Initialize the SQLite database:

```bash
python scripts/init_db.py
```

5. Run the app:

```bash
python -m app.main
```

## Quickstart (Google Colab)

1. Upload your project folder (or clone your repo) in Colab.
2. Install OCR system dependency:

```python
!apt-get -y update
!apt-get install -y tesseract-ocr
```

3. Install Python dependencies:

```python
!pip install -r requirements.txt
```

4. Initialize the SQLite database:

```python
!python scripts/init_db.py
```

5. Run the app entrypoint:

```python
!python -m app.main
```

## Example Queries (MVP)

1. Upload receipt and process:
  - "Process this receipt"
2. List receipts:
  - "List receipts"
3. Spend by vendor:
  - "Show spend by vendor"
4. Find duplicates:
  - "Find duplicates"
