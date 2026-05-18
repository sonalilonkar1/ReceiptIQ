# ReceiptIQ - Complete Setup Guide

Complete step-by-step instructions to set up ReceiptIQ on your machine.

**Estimated time: 15-30 minutes** (depending on internet speed and system specs)

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Checklist](#pre-installation-checklist)
3. [Installation Steps](#installation-steps)
4. [Verification](#verification)
5. [Optional Enhancements](#optional-enhancements)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **Processor:** Intel/AMD x64 or Apple Silicon (M1/M2/M3)
- **RAM:** 8GB minimum, 16GB+ recommended (for LLM inference)
- **Disk Space:** 5GB free (includes models and datasets)
- **GPU:** Optional but recommended for faster LLM inference (NVIDIA CUDA 11.8+ or Apple Silicon)

### Software
- **OS:** macOS 11+, Ubuntu 18.04+, Debian 11+, or Windows 10/11 (with WSL2)
- **Python:** 3.9, 3.10, 3.11, or 3.12
- **Git:** For cloning repository (optional)
- **Internet:** Required for downloading models and datasets

---

## Pre-Installation Checklist

### Check Your Python Version

```bash
python3 --version
# Output should be: Python 3.9.x, 3.10.x, 3.11.x, or 3.12.x

# If you have multiple Python versions, use specific version:
python3.11 --version
```

If you don't have Python 3.9+, download from [python.org](https://www.python.org/downloads/)

### Check Your Package Manager

#### macOS
```bash
# Check if Homebrew is installed
brew --version

# If not, install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Ubuntu/Debian
```bash
# Check if apt is available
apt --version

# Update package list
sudo apt-get update
```

#### Windows (WSL2)
```bash
# Open Windows Terminal and run:
wsl --list --verbose
# Output should show Ubuntu or Debian with "Running" status
```

---

## Installation Steps

### Step 1: Install System Dependencies

#### macOS
```bash
brew install tesseract poppler python@3.11 git
```

Verify:
```bash
tesseract --version
pdftoppm --version
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip
sudo apt-get install -y tesseract-ocr poppler-utils git
sudo apt-get install -y libopencv-dev
```

Verify:
```bash
tesseract --version
pdftoppm --version
python3.11 --version
```

#### Windows (WSL2)
```bash
# Open WSL2 terminal and run:
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip
sudo apt-get install -y tesseract-ocr poppler-utils git
sudo apt-get install -y libopencv-dev
```

### Step 2: Clone/Navigate to Project

```bash
# Option A: Clone repository (if not already done)
git clone https://github.com/your-repo/ReceiptIQ.git
cd ReceiptIQ

# Option B: Navigate to existing project
cd /path/to/ReceiptIQ
pwd  # Verify location
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
# macOS/Linux/WSL2:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Verify activation (should show .venv in your prompt)
which python    # macOS/Linux
where python    # Windows
```

**💡 Tip:** Always activate `.venv` before working on the project:
```bash
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### Step 4: Upgrade pip and Install Dependencies

```bash
# Ensure pip is up to date
pip install --upgrade pip setuptools wheel

# Install all project dependencies
pip install -r requirements.txt

# This installs:
# - PyTorch (ML framework)
# - Transformers (HuggingFace models)
# - Gradio (web UI)
# - Pillow (image processing)
# - pytesseract (OCR interface)
# - And 15+ other packages
```

**⏱️ Expected time: 3-5 minutes** (or 10+ min on slow internet)

Verify installation:
```bash
python -c "import torch; import transformers; import gradio; print('✅ All core packages installed')"
```

### Step 5: Initialize Database

```bash
# Create SQLite database with schema
python scripts/init_db.py

# Output should show:
# ✓ Database initialized at data/receipts_db/receipts.json
# ✓ 5 tables created: documents, audit_flags, expense_rules, reimbursement_batches, batch_documents
```

Verify database:
```bash
ls -lh data/receipts_db/
# Should show: receipts.json (SQLite database file)
```

### Step 6: Download Models & Datasets (Optional but Recommended)

```bash
# Download Donut OCR-free model (~1.5GB)
# This enables extraction fallback when Tesseract fails
python -c "from transformers import DonutProcessor; DonutProcessor.from_pretrained('hf-tuner/donut-base-finetuned-sroie'); print('✅ Donut model downloaded')"

# Download SROIE dataset for benchmarking (100 samples, ~50MB)
python scripts/download_sroie_100.py --n 100 --split train --out_dir data/sroie_100

# Download CORD dataset for testing (100 samples, ~50MB)
python scripts/download_cord_subset.py --n 100 --out_dir data/cord_100
```

**⏱️ Expected time: 5-10 minutes** on first run

---

## Verification

### Quick Smoke Test

```bash
# Run rapid health check
python scripts/smoke_test.py

# Expected output:
# ✅ All systems operational
# ✅ Database accessible
# ✅ Models loaded successfully
```

### Run All Tests (Optional)

```bash
# Security tests (6 tests)
python test_security_guard.py
# Expected: 6/6 passed ✓

# Guardrail tests (23 tests)
python scripts/run_guardrail_checks.py
# Expected: 23/23 passed ✓
```

### Launch Application

```bash
# Start the web interface
python app/main.py

# Output should show:
# Running on http://localhost:7860
# To create a public link, set `share=True` in `launch()`.
```

**🎉 Success!** Open browser to `http://localhost:7860`

---

## Optional Enhancements

### A. Enable Ollama for Local LLM Inference

**What it does:** Run Phi-3.5-mini or Mistral locally for faster inference

#### Install Ollama

```bash
# macOS
brew install ollama

# Or download from: https://ollama.ai/download

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Download Models

```bash
# Start Ollama service in background
ollama serve &

# Download Phi-3.5-mini (fast, good quality)
ollama pull phi

# Or Mistral-7B (slower, higher quality)
ollama pull mistral

# List downloaded models
ollama list
```

#### Configure Project

Edit `app/agent.py`:
```python
USE_OLLAMA = True
MODEL_MODE = "phi_only"  # or "phi+mistral"
OLLAMA_ENDPOINT = "http://localhost:11434"
```

Edit `app/tools/llm_parser.py`:
```python
OLLAMA_TIMEOUT = 120  # seconds
```

#### Verify

```bash
python -c "
import requests
try:
    r = requests.get('http://localhost:11434/api/tags', timeout=2)
    print('✅ Ollama is running') if r.status_code == 200 else print('❌ Ollama error')
except:
    print('❌ Ollama not running. Start with: ollama serve')
"
```

### B. Setup Environment Variables

Create `.env` file in project root:

```bash
# LLM Configuration
RECEIPTIQ_OLLAMA_TIMEOUT=120
RECEIPTIQ_OLLAMA_ENDPOINT=http://localhost:11434
RECEIPTIQ_MODEL_MODE=phi_only
RECEIPTIQ_USE_PROMPT_CACHE=true

# Database
RECEIPTIQ_DB_PATH=data/receipts_db

# OCR
RECEIPTIQ_OCR_LANGUAGE=eng
RECEIPTIQ_TESSERACT_PATH=/usr/bin/tesseract
```

Load in Python:
```python
from dotenv import load_dotenv
import os
load_dotenv()
timeout = os.getenv("RECEIPTIQ_OLLAMA_TIMEOUT", "120")
```

### C. Setup GPU Support (Optional)

#### NVIDIA GPUs (CUDA)

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-compatible PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Apple Silicon (M1/M2/M3)

```bash
# PyTorch automatically uses Metal Performance Shaders
# Just verify:
python -c "import torch; print(torch.backends.mps.is_available())"
```

### D. Load Sample Receipts to Database

```bash
# Load SROIE receipts to database for testing
python scripts/load_sroie_to_db.py --limit 10

# Verify in application: Upload a receipt or query "Show recent receipts"
```

---

## Troubleshooting

### Common Issues & Solutions

#### Issue: `ModuleNotFoundError: No module named 'torch'`

**Cause:** Dependencies not installed or virtual environment not activated

**Solution:**
```bash
# Make sure you're in the project directory
cd /path/to/ReceiptIQ

# Activate virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

#### Issue: `tesseract-ocr is not installed on this system`

**Cause:** System-level OCR library not installed

**Solution:**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows (WSL2)
sudo apt-get install tesseract-ocr
```

Verify:
```bash
tesseract --version
```

---

#### Issue: `Port 7860 already in use`

**Cause:** Another application is using the port

**Solution:**
```bash
# Option 1: Use different port
python app/main.py --port 7861

# Option 2: Kill existing process (macOS/Linux)
lsof -i :7860
kill -9 <PID>
```

---

#### Issue: `requests.exceptions.ConnectionError: [Errno -2] Name or service not known`

**Cause:** Ollama not running or misconfigured

**Solution:**
```bash
# Start Ollama in a new terminal
ollama serve

# Verify in another terminal
curl http://localhost:11434/api/tags

# Or disable Ollama in app/agent.py
USE_OLLAMA = False
```

---

#### Issue: `Out of memory` or slow inference

**Cause:** System doesn't have enough RAM for large models

**Solution:**
```python
# Option 1: Use smaller model
# In app/agent.py:
MODEL_MODE = "phi_only"  # Instead of phi+mistral

# Option 2: Use Google Colab (free GPU)
# Open ReceiptIQ_Colab.ipynb

# Option 3: Reduce batch size (if applicable)
# Disable prompt caching for memory savings:
PROMPT_CACHE_ENABLED = False
```

---

#### Issue: `ModuleNotFoundError: No module named 'dotenv'`

**Cause:** python-dotenv package not installed

**Solution:**
```bash
pip install python-dotenv
```

---

#### Issue: GPU not detected (`torch.cuda.is_available()` returns False)

**Cause:** CUDA not installed or PyTorch installed without GPU support

**Solution:**
```bash
# For NVIDIA GPUs, install CUDA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon, PyTorch should auto-detect
# Verify: python -c "import torch; print(torch.backends.mps.is_available())"
```

---

### Getting Help

If you encounter issues:

1. **Check [TROUBLESHOOTING.md](../Other/docs/TROUBLESHOOTING.md)** — 30+ detailed scenarios
2. **Run diagnostic script:**
   ```bash
   python scripts/smoke_test.py
   ```
3. **Check logs:**
   ```bash
   # Look for error messages in terminal output
   # Enable debug mode in UI: "Pending Receipts" → "Enable Debug Logs"
   ```

---

## Next Steps After Setup

1. **Explore the UI** — Open `http://localhost:7860` and upload a receipt
2. **Run tests** — Verify everything works: `python scripts/run_guardrail_checks.py`
3. **Read documentation** — Check [../Other/docs/README.md](../Other/docs/README.md)
4. **Try example queries** — See [README.md#example-prompts](README.md#example-prompts)

---

## Uninstall / Cleanup

If you want to remove ReceiptIQ:

```bash
# Delete virtual environment
rm -rf .venv

# Delete downloaded models (optional, reclaims ~3GB)
rm -rf ~/.cache/huggingface/

# Delete project directory
cd ..
rm -rf ReceiptIQ
```

---

## Support

- **Documentation:** [../Other/docs/README.md](../Other/docs/README.md)
- **Troubleshooting:** [../Other/docs/TROUBLESHOOTING.md](../Other/docs/TROUBLESHOOTING.md)
- **Architecture:** [../Other/docs/PROJECT_ARCHITECTURE.md](../Other/docs/PROJECT_ARCHITECTURE.md)
- **API Reference:** [../Other/docs/API_REFERENCE.md](../Other/docs/API_REFERENCE.md)
- **Deployment:** [../Other/docs/DEPLOYMENT_GUIDE.md](../Other/docs/DEPLOYMENT_GUIDE.md)
