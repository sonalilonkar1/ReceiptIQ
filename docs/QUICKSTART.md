# ReceiptIQ - Quick Start (5 minutes)

**The fastest way to get ReceiptIQ running.**

---

## Copy-Paste Commands

```bash
# 1. Clone or navigate to project
cd /path/to/ReceiptIQ

# 2. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Initialize database
python scripts/init_db.py

# 5. Launch application
python app/main.py

# 6. Open browser
# → http://localhost:7860
```

**Done!** ✅ The application is now running.

---

## Try It Out

### Upload a Receipt
1. Click **Chat** tab
2. Click **"Upload a receipt"**
3. Select an image of a receipt

### Analyze Spending
```
"Weekly summary"
"Monthly summary"
"Show spending by vendor"
```

### Complete Pending Receipts
1. Click **Pending Receipts** tab
2. Select a receipt
3. Fill in missing information
4. Click **Update Receipt**

---

## Common Commands

```bash
# Activate environment (do this in every new terminal)
source .venv/bin/activate

# Run application
python app/main.py

# Run all tests
python scripts/run_guardrail_checks.py

# Run security tests
python test_security_guard.py

# Run benchmarks
python scripts/run_benchmark.py
```

---

## Troubleshooting (3 Quick Fixes)

### ❌ "Port 7860 already in use"
```bash
python app/main.py --port 7861
```

### ❌ "ModuleNotFoundError: torch"
```bash
pip install -r requirements.txt
```

### ❌ "tesseract-ocr is not installed"
```bash
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
```

---

## Need More Help?

- **Detailed setup:** See [SETUP.md](SETUP.md)
- **Full documentation:** See [README.md](README.md)
- **Troubleshooting:** See [../Other/docs/TROUBLESHOOTING.md](../Other/docs/TROUBLESHOOTING.md)

---

## Optional: Run with Ollama (LLM)

```bash
# 1. Install Ollama
brew install ollama  # macOS

# 2. Download model (in new terminal)
ollama pull phi

# 3. Start Ollama service
ollama serve &

# 4. Edit app/agent.py:
USE_OLLAMA = True

# 5. Run application
python app/main.py
```

**That's it!** You're ready to go. 🚀
